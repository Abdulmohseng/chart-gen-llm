import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import io
import base64
from langchain.agents import tool
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_ollama import ChatOllama

# Initialize styling for charts
plt.style.use('ggplot')
sns.set_palette("deep")

class Dataset:
    """Class to hold and analyze a dataset"""
    
    def __init__(self, df=None):
        self.df = df
        self.summary = {}
        self.column_types = {}
        
    def load_csv(self, file_path):
        """Load a CSV file into the dataset"""
        try:
            self.df = pd.read_csv(file_path)
            self._analyze_dataset()
            return True, f"Successfully loaded dataset with {self.df.shape[0]} rows and {self.df.shape[1]} columns."
        except Exception as e:
            return False, f"Error loading CSV: {str(e)}"
    
    def _analyze_dataset(self):
        """Analyze the dataset and extract key information"""
        if self.df is None:
            return
        
        # Get basic info
        self.summary["rows"] = self.df.shape[0]
        self.summary["columns"] = self.df.shape[1]
        self.summary["column_names"] = self.df.columns.tolist()
        self.summary["missing_values"] = self.df.isna().sum().to_dict()
        
        # Analyze column types
        self.column_types = {}
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if self.df[col].nunique() <= 10:
                    self.column_types[col] = "categorical_numeric"
                else:
                    self.column_types[col] = "continuous"
            elif pd.api.types.is_datetime64_dtype(self.df[col]):
                self.column_types[col] = "datetime"
            else:
                if self.df[col].nunique() <= 20:
                    self.column_types[col] = "categorical"
                else:
                    self.column_types[col] = "text"
                    
        # Additional statistics
        self.summary["categorical_columns"] = [col for col, type_ in self.column_types.items() 
                                              if type_ in ["categorical", "categorical_numeric"]]
        self.summary["numeric_columns"] = [col for col, type_ in self.column_types.items() 
                                          if type_ in ["continuous", "categorical_numeric"]]
        self.summary["datetime_columns"] = [col for col, type_ in self.column_types.items() 
                                           if type_ == "datetime"]
        
        # Get basic statistics for numeric columns
        if self.summary["numeric_columns"]:
            self.summary["numeric_stats"] = self.df[self.summary["numeric_columns"]].describe().to_dict()
        
        # Get value counts for categorical columns (limited to top 5)
        self.summary["categorical_value_counts"] = {}
        for col in self.summary["categorical_columns"]:
            self.summary["categorical_value_counts"][col] = self.df[col].value_counts().head(5).to_dict()
            
    def get_summary(self):
        """Return a summary of the dataset"""
        if self.df is None:
            return "No dataset loaded."
        
        return self.summary
    
    def get_column_types(self):
        """Return the detected column types"""
        if self.df is None:
            return "No dataset loaded."
        
        return self.column_types

# Initialize the dataset object
dataset = Dataset()

@tool
def load_dataset(file_path: str) -> str:
    """
    Load a dataset from a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        A message indicating if the dataset was loaded successfully.
    """
    success, message = dataset.load_csv(file_path)
    return message

@tool
def analyze_dataset() -> Dict:
    """
    Analyze the current dataset and return a summary of its properties.
    
    Returns:
        A dictionary containing summary statistics about the dataset.
    """
    if dataset.df is None:
        return "No dataset has been loaded. Please load a dataset first."
    
    return dataset.get_summary()

class ChartRecommendation(BaseModel):
    chart_type: str = Field(description="The type of chart to create")
    title: str = Field(description="A descriptive title for the chart")
    description: str = Field(description="A brief explanation of why this chart is appropriate")
    x_column: str = Field(description="Column to use for x-axis")
    y_column: str = Field(description="Column to use for y-axis (leave empty for pie charts)")
    additional_params: Dict[str, Any] = Field(description="Additional parameters for the chart")

@tool
def recommend_charts() -> List[Dict]:
    """
    Recommend appropriate chart types based on the dataset analysis.
    
    Returns:
        A list of recommended chart types with explanations.
    """
    if dataset.df is None:
        return "No dataset has been loaded. Please load a dataset first."
    
    column_types = dataset.get_column_types()
    summary = dataset.get_summary()
    
    chart_prompt = PromptTemplate.from_template("""
    You are a data visualization expert. Based on the following dataset analysis, 
    recommend 3-5 appropriate chart types that would provide meaningful insights.
    
    Dataset summary: {summary}
    
    Column types: {column_types}
    
    For each recommendation, provide:
    - Chart type (e.g., bar chart, scatter plot, line chart)
    - A specific title that reflects what the chart will show
    - A brief explanation of why this chart is appropriate
    - Which specific columns should be used (x-axis and y-axis where applicable)
    - Any additional parameters that should be set
    
    Only recommend charts that are appropriate for the data. For example:
    - Scatter plots require two numerical columns
    - Pie charts should only be used for categorical data with few categories
    - Line charts are best for time series or continuous data
    - Bar charts work well for categorical comparisons
    
    Return your recommendations as a JSON array.
    """)
    
    # Create a JSON output parser
    class ChartRecommendations(BaseModel):
        recommendations: List[ChartRecommendation] = Field(description="List of chart recommendations")
    
    parser = JsonOutputParser(pydantic_object=ChartRecommendations)
    
    # Create an LLM to generate recommendations
    llm = ChatOpenAI(temperature=0.1)
    chain = chart_prompt | llm | parser
    
    response = chain.invoke({
        "summary": summary,
        "column_types": column_types
    })
    
    return response["recommendations"]

@tool
def validate_chart(chart_type: str, x_column: str, y_column: str = None, additional_params: Dict[str, Any] = None) -> str:
    """
    Validate if a chart type is appropriate for the selected columns.
    
    Args:
        chart_type: The type of chart to validate.
        x_column: Column to use for x-axis.
        y_column: Column to use for y-axis (optional for some chart types).
        additional_params: Additional parameters for the chart.
        
    Returns:
        A message indicating if the chart type is valid for the selected columns.
    """
    if dataset.df is None:
        return "No dataset has been loaded. Please load a dataset first."
    
    if x_column not in dataset.df.columns:
        return f"Error: Column '{x_column}' not found in the dataset."
        
    if y_column and y_column not in dataset.df.columns:
        return f"Error: Column '{y_column}' not found in the dataset."
    
    column_types = dataset.get_column_types()
    
    # Validation logic for different chart types
    if chart_type.lower() in ["scatter", "scatter plot"]:
        if not y_column:
            return "Error: Scatter plots require both x and y columns."
        if column_types[x_column] not in ["continuous", "categorical_numeric"]:
            return f"Warning: Scatter plots work best with numeric x values, but '{x_column}' is {column_types[x_column]}."
        if column_types[y_column] not in ["continuous", "categorical_numeric"]:
            return f"Warning: Scatter plots work best with numeric y values, but '{y_column}' is {column_types[y_column]}."
        return "Valid: Scatter plot is appropriate for these columns."
    
    elif chart_type.lower() in ["bar", "bar chart"]:
        if not y_column:
            return "Error: Bar charts require both x and y columns."
        if column_types[x_column] not in ["categorical", "categorical_numeric"]:
            return f"Warning: Bar charts work best with categorical x values, but '{x_column}' is {column_types[x_column]}."
        if column_types[y_column] not in ["continuous", "categorical_numeric"]:
            return f"Warning: Bar charts work best with numeric y values, but '{y_column}' is {column_types[y_column]}."
        return "Valid: Bar chart is appropriate for these columns."
    
    elif chart_type.lower() in ["line", "line chart"]:
        if not y_column:
            return "Error: Line charts require both x and y columns."
        if column_types[y_column] not in ["continuous", "categorical_numeric"]:
            return f"Warning: Line charts work best with numeric y values, but '{y_column}' is {column_types[y_column]}."
        return "Valid: Line chart is appropriate for these columns."
    
    elif chart_type.lower() in ["pie", "pie chart"]:
        if y_column:
            return "Note: Pie charts typically use only one column to determine categories. The second column will be used for values."
        if column_types[x_column] not in ["categorical", "categorical_numeric"]:
            return f"Warning: Pie charts work best with categorical data, but '{x_column}' is {column_types[x_column]}."
        if dataset.df[x_column].nunique() > 10:
            return f"Warning: Pie charts work best with a small number of categories, but '{x_column}' has {dataset.df[x_column].nunique()} unique values."
        return "Valid: Pie chart is appropriate for this column."
    
    elif chart_type.lower() in ["histogram"]:
        if y_column:
            return "Note: Histograms typically only need one numeric column. The second column will be ignored."
        if column_types[x_column] not in ["continuous", "categorical_numeric"]:
            return f"Warning: Histograms work best with numeric data, but '{x_column}' is {column_types[x_column]}."
        return "Valid: Histogram is appropriate for this column."
    
    elif chart_type.lower() in ["boxplot", "box plot"]:
        if not y_column:
            return "Note: Box plots can use one numeric column, but specifying a categorical column for x helps group the data."
        if column_types[y_column] not in ["continuous", "categorical_numeric"]:
            return f"Warning: Box plots work best with numeric y values, but '{y_column}' is {column_types[y_column]}."
        return "Valid: Box plot is appropriate for these columns."
    
    elif chart_type.lower() in ["heatmap", "heat map"]:
        if not y_column:
            return "Error: Heatmaps require both x and y columns."
        if dataset.df.shape[1] < 3:
            return "Error: Heatmaps typically need at least 3 columns (x, y, and a value column)."
        return "Valid: Heatmap is appropriate for these columns."
    
    else:
        return f"Unknown chart type: {chart_type}"

def _figure_to_base64(fig):
    """Convert a matplotlib figure to a base64 encoded string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

@tool
def generate_chart_code(chart_type: str, x_column: str, y_column: str = None, 
                        title: str = None, additional_params: Dict[str, Any] = None) -> str:
    """
    Generate Python code to create the specified chart.
    
    Args:
        chart_type: The type of chart to create.
        x_column: Column to use for x-axis.
        y_column: Column to use for y-axis (optional for some chart types).
        title: Title for the chart.
        additional_params: Additional parameters for the chart.
        
    Returns:
        Python code to generate the chart.
    """
    if dataset.df is None:
        return "No dataset has been loaded. Please load a dataset first."
    
    if not title:
        title = f"{chart_type.title()} of {x_column}"
        if y_column:
            title += f" vs {y_column}"
    
    if not additional_params:
        additional_params = {}
    
    # Create code based on chart type
    code = f"import matplotlib.pyplot as plt\nimport seaborn as sns\n\n"
    code += f"# Create a {chart_type} chart\n"
    code += f"plt.figure(figsize=(10, 6))\n"
    
    if chart_type.lower() in ["scatter", "scatter plot"]:
        code += f"plt.scatter(df['{x_column}'], df['{y_column}']"
        if 'color' in additional_params:
            code += f", color='{additional_params['color']}'"
        if 'alpha' in additional_params:
            code += f", alpha={additional_params['alpha']}"
        code += ")\n"
        code += f"plt.xlabel('{x_column}')\n"
        code += f"plt.ylabel('{y_column}')\n"
    
    elif chart_type.lower() in ["bar", "bar chart"]:
        code += f"sns.barplot(x='{x_column}', y='{y_column}', data=df"
        if 'hue' in additional_params:
            code += f", hue='{additional_params['hue']}'"
        code += ")\n"
        code += f"plt.xlabel('{x_column}')\n"
        code += f"plt.ylabel('{y_column}')\n"
        if 'rotate_xticks' in additional_params and additional_params['rotate_xticks']:
            code += "plt.xticks(rotation=45, ha='right')\n"
    
    elif chart_type.lower() in ["line", "line chart"]:
        if 'hue' in additional_params:
            code += f"sns.lineplot(x='{x_column}', y='{y_column}', hue='{additional_params['hue']}', data=df)\n"
        else:
            code += f"plt.plot(df['{x_column}'], df['{y_column}'])\n"
        code += f"plt.xlabel('{x_column}')\n"
        code += f"plt.ylabel('{y_column}')\n"
    
    elif chart_type.lower() in ["pie", "pie chart"]:
        if y_column:
            code += f"values = df.groupby('{x_column}')['{y_column}'].sum()\n"
            code += f"plt.pie(values, labels=values.index, autopct='%1.1f%%'"
        else:
            code += f"value_counts = df['{x_column}'].value_counts()\n"
            code += f"plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%'"
        if 'startangle' in additional_params:
            code += f", startangle={additional_params['startangle']}"
        code += ")\n"
    
    elif chart_type.lower() in ["histogram"]:
        code += f"plt.hist(df['{x_column}']"
        if 'bins' in additional_params:
            code += f", bins={additional_params['bins']}"
        if 'color' in additional_params:
            code += f", color='{additional_params['color']}'"
        code += ")\n"
        code += f"plt.xlabel('{x_column}')\n"
        code += f"plt.ylabel('Frequency')\n"
    
    elif chart_type.lower() in ["boxplot", "box plot"]:
        if y_column:
            code += f"sns.boxplot(x='{x_column}', y='{y_column}', data=df"
        else:
            code += f"sns.boxplot(y='{x_column}', data=df"
        if 'hue' in additional_params:
            code += f", hue='{additional_params['hue']}'"
        code += ")\n"
    
    elif chart_type.lower() in ["heatmap", "heat map"]:
        if 'value_column' in additional_params:
            value_col = additional_params['value_column']
            code += f"pivot_table = df.pivot_table(index='{y_column}', columns='{x_column}', values='{value_col}')\n"
        else:
            code += f"pivot_table = df.pivot_table(index='{y_column}', columns='{x_column}')\n"
        code += f"sns.heatmap(pivot_table, annot=True, cmap='viridis')\n"
    
    else:
        return f"Unsupported chart type: {chart_type}"
    
    code += f"plt.title('{title}')\n"
    code += "plt.tight_layout()\n"
    code += "plt.show()\n"
    
    return code

@tool
def create_chart(chart_type: str, x_column: str, y_column: str = None, 
                title: str = None, additional_params: Dict[str, Any] = None) -> Dict:
    """
    Create and return a chart based on the specified parameters.
    
    Args:
        chart_type: The type of chart to create.
        x_column: Column to use for x-axis.
        y_column: Column to use for y-axis (optional for some chart types).
        title: Title for the chart.
        additional_params: Additional parameters for the chart.
        
    Returns:
        A dictionary containing the chart's code and base64-encoded image.
    """
    if dataset.df is None:
        return {"error": "No dataset has been loaded. Please load a dataset first."}
    
    if not title:
        title = f"{chart_type.title()} of {x_column}"
        if y_column:
            title += f" vs {y_column}"
    
    if not additional_params:
        additional_params = {}
    
    df = dataset.df
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if chart_type.lower() in ["scatter", "scatter plot"]:
            ax.scatter(df[x_column], df[y_column],
                     color=additional_params.get('color', 'blue'),
                     alpha=additional_params.get('alpha', 0.7))
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
        
        elif chart_type.lower() in ["bar", "bar chart"]:
            if 'hue' in additional_params:
                sns.barplot(x=x_column, y=y_column, hue=additional_params['hue'], data=df, ax=ax)
            else:
                sns.barplot(x=x_column, y=y_column, data=df, ax=ax)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            if additional_params.get('rotate_xticks', False):
                plt.xticks(rotation=45, ha='right')
        
        elif chart_type.lower() in ["line", "line chart"]:
            if 'hue' in additional_params:
                sns.lineplot(x=x_column, y=y_column, hue=additional_params['hue'], data=df, ax=ax)
            else:
                ax.plot(df[x_column], df[y_column])
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
        
        elif chart_type.lower() in ["pie", "pie chart"]:
            if y_column:
                values = df.groupby(x_column)[y_column].sum()
                ax.pie(values, labels=values.index, autopct='%1.1f%%',
                      startangle=additional_params.get('startangle', 90))
            else:
                value_counts = df[x_column].value_counts()
                ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',
                      startangle=additional_params.get('startangle', 90))
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        elif chart_type.lower() in ["histogram"]:
            ax.hist(df[x_column],
                  bins=additional_params.get('bins', 10),
                  color=additional_params.get('color', 'blue'))
            ax.set_xlabel(x_column)
            ax.set_ylabel('Frequency')
        
        elif chart_type.lower() in ["boxplot", "box plot"]:
            if y_column:
                if 'hue' in additional_params:
                    sns.boxplot(x=x_column, y=y_column, hue=additional_params['hue'], data=df, ax=ax)
                else:
                    sns.boxplot(x=x_column, y=y_column, data=df, ax=ax)
            else:
                sns.boxplot(y=x_column, data=df, ax=ax)
        
        elif chart_type.lower() in ["heatmap", "heat map"]:
            if 'value_column' in additional_params:
                value_col = additional_params['value_column']
                pivot_table = df.pivot_table(index=y_column, columns=x_column, values=value_col)
            else:
                # Try to use the first numeric column that's not x or y for values
                numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                numeric_cols = [col for col in numeric_cols if col != x_column and col != y_column]
                
                if not numeric_cols:
                    return {"error": "No numeric column available for heatmap values. Please specify a value_column in additional_params."}
                    
                value_col = numeric_cols[0]
                pivot_table = df.pivot_table(index=y_column, columns=x_column, values=value_col)
            
            sns.heatmap(pivot_table, annot=True, cmap='viridis', ax=ax)
        
        else:
            return {"error": f"Unsupported chart type: {chart_type}"}
        
        ax.set_title(title)
        plt.tight_layout()
        
        # Convert the figure to base64
        img_str = _figure_to_base64(fig)
        
        # Generate code
        code = generate_chart_code(chart_type, x_column, y_column, title, additional_params)
        
        return {
            "code": code,
            "image": img_str
        }
    
    except Exception as e:
        return {"error": f"Error creating chart: {str(e)}"}

@tool
def enhance_chart(chart_type: str, x_column: str, y_column: str = None, 
                 title: str = None, enhancements: Dict[str, Any] = None) -> Dict:
    """
    Create an enhanced chart with additional formatting and features.
    
    Args:
        chart_type: The type of chart to create.
        x_column: Column to use for x-axis.
        y_column: Column to use for y-axis (optional for some chart types).
        title: Title for the chart.
        enhancements: Dictionary of enhancements to apply to the chart.
            Can include: color_palette, add_trendline, rotate_labels, add_annotations,
            highlight_values, log_scale, grid, legend_position, etc.
        
    Returns:
        A dictionary containing the enhanced chart's code and base64-encoded image.
    """
    if dataset.df is None:
        return {"error": "No dataset has been loaded. Please load a dataset first."}
    
    if not enhancements:
        enhancements = {}
    
    df = dataset.df
    
    try:
        fig, ax = plt.subplots(figsize=enhancements.get('figsize', (10, 6)))
        
        # Apply color palette if specified
        if 'color_palette' in enhancements:
            sns.set_palette(enhancements['color_palette'])
        
        # Create base chart
        if chart_type.lower() in ["scatter", "scatter plot"]:
            scatter = ax.scatter(df[x_column], df[y_column],
                               alpha=enhancements.get('alpha', 0.7),
                               s=enhancements.get('marker_size', 50))
            
            # Add trendline if requested
            if enhancements.get('add_trendline', False):
                import numpy as np
                from scipy import stats
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_column], df[y_column])
                x_line = np.array([df[x_column].min(), df[x_column].max()])
                y_line = intercept + slope * x_line
                ax.plot(x_line, y_line, color='red', linestyle='--', 
                      label=f'Trend line (r²={r_value**2:.2f})')
                ax.legend()
        
        elif chart_type.lower() in ["bar", "bar chart"]:
            if 'hue' in enhancements:
                bars = sns.barplot(x=x_column, y=y_column, hue=enhancements['hue'], data=df, ax=ax)
            else:
                bars = sns.barplot(x=x_column, y=y_column, data=df, ax=ax)
            
            # Add value labels on top of bars
            if enhancements.get('add_value_labels', False):
                for p in bars.patches:
                    bars.annotate(f"{p.get_height():.1f}", 
                                 (p.get_x() + p.get_width() / 2., p.get_height()),
                                 ha='center', va='bottom', fontsize=10)
        
        elif chart_type.lower() in ["line", "line chart"]:
            if 'hue' in enhancements:
                lines = sns.lineplot(x=x_column, y=y_column, hue=enhancements['hue'], 
                                   marker=enhancements.get('marker', 'o'), 
                                   data=df, ax=ax)
            else:
                lines = ax.plot(df[x_column], df[y_column], 
                              marker=enhancements.get('marker', 'o'))
            
            # Add area fill under the line
            if enhancements.get('fill_area', False) and 'hue' not in enhancements:
                ax.fill_between(df[x_column], df[y_column], alpha=0.2)
        
        elif chart_type.lower() in ["pie", "pie chart"]:
            if y_column:
                values = df.groupby(x_column)[y_column].sum()
            else:
                values = df[x_column].value_counts()
            
            # Explode slices if specified
            if 'explode' in enhancements:
                explode = [0.1] * len(values)
            else:
                explode = None
                
            wedges, texts, autotexts = ax.pie(
                values, 
                labels=values.index if not enhancements.get('remove_labels', False) else None,
                autopct='%1.1f%%' if enhancements.get('show_percentages', True) else None,
                shadow=enhancements.get('shadow', False),
                explode=explode,
                startangle=enhancements.get('startangle', 90)
            )
            
            # Customize text appearance
            if enhancements.get('show_percentages', True):
                for autotext in autotexts:
                    autotext.set_color(enhancements.get('percent_text_color', 'white'))
                    autotext.set_fontsize(enhancements.get('percent_text_size', 10))
            
            ax.axis('equal')
        
        elif chart_type.lower() in ["histogram"]:
            hist = ax.hist(df[x_column],
                         bins=enhancements.get('bins', 10),
                         color=enhancements.get('color', 'blue'),
                         alpha=enhancements.get('alpha', 0.7),
                         edgecolor=enhancements.get('edge_color', 'black'))
            
            # Add KDE curve
            if enhancements.get('add_kde', False):
                from scipy import stats
                import numpy as np
                
                kde_xs = np.linspace(df[x_column].min(), df[x_column].max(), 100)
                kde = stats.gaussian_kde(df[x_column].dropna())
                kde_ys = kde(kde_xs)
                
                # Scale KDE to match histogram height
                hist_height = hist[0].max()
                kde_height = kde_ys.max()
                scaling_factor = hist_height / kde_height
                
                twin_ax = ax.twinx()
                twin_ax.plot(kde_xs, kde_ys * scaling_factor, 'r-', linewidth=2)
                twin_ax.set_yticks([])
        
        elif chart_type.lower() in ["boxplot", "box plot"]:
            if y_column:
                if 'hue' in enhancements:
                    sns.boxplot(x=x_column, y=y_column, hue=enhancements['hue'], 
                              data=df, ax=ax, 
                              width=enhancements.get('width', 0.8))
                else:
                    sns.boxplot(x=x_column, y=y_column, data=df, ax=ax)
                
                # Add swarmplot overlay
                if enhancements.get('add_swarmplot', False):
                    sns.swarmplot(x=x_column, y=y_column, data=df, ax=ax, 
                                color=enhancements.get('swarm_color', '.25'),
                                size=enhancements.get('swarm_size', 5),
                                alpha=enhancements.get('swarm_alpha', 0.5))
            else:
                sns.boxplot(y=x_column, data=df, ax=ax)
                
                # Add swarmplot overlay
                if enhancements.get('add_swarmplot', False):
                    sns.swarmplot(y=x_column, data=df, ax=ax, 
                                color=enhancements.get('swarm_color', '.25'),
                                size=enhancements.get('swarm_size', 5),
                                alpha=enhancements.get('swarm_alpha', 0.5))
        
        elif chart_type.lower() in ["heatmap", "heat map"]:
            if 'value_column' in enhancements:
                value_col = enhancements['value_column']
            else:
                # Try to use the first numeric column that's not x or y for values
                numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                numeric_cols = [col for col in numeric_cols if col != x_column and col != y_column]
                
                if not numeric_cols:
                    return {"error": "No numeric column available for heatmap values. Please specify a value_column in enhancements."}
                    
                value_col = numeric_cols[0]
            
            pivot_table = df.pivot_table(index=y_column, columns=x_column, values=value_col)
            
            sns.heatmap(pivot_table, 
                      annot=enhancements.get('show_values', True),
                      cmap=enhancements.get('colormap', 'viridis'),
                      linewidths=enhancements.get('linewidths', 0.5),
                      fmt=enhancements.get('value_format', '.2f'),
                      ax=ax)
        
        else:
            return {"error": f"Unsupported chart type: {chart_type}"}
        
        # Apply common enhancements
        
        # Set title with custom size
        if title:
            title_fontsize = enhancements.get('title_fontsize', 14)
            ax.set_title(title, fontsize=title_fontsize, fontweight='bold' if enhancements.get('bold_title', False) else 'normal')
        
        # Set axis labels with custom size
        if 'x_label' in enhancements:
            ax.set_xlabel(enhancements['x_label'], fontsize=enhancements.get('label_fontsize', 12))
        else:
            ax.set_xlabel(x_column, fontsize=enhancements.get('label_fontsize', 12))
            
        if 'y_label' in enhancements and chart_type.lower() not in ['pie', 'pie chart']:
            ax.set_ylabel(enhancements['y_label'], fontsize=enhancements.get('label_fontsize', 12))
        elif y_column and chart_type.lower() not in ['pie', 'pie chart']:
            ax.set_ylabel(y_column, fontsize=enhancements.get('label_fontsize', 12))
            
        # Rotate x-axis labels
        if enhancements.get('rotate_xticks', False):
            plt.xticks(rotation=enhancements.get('xtick_rotation', 45), ha='right')
            
        # Apply log scale
        if enhancements.get('log_scale_y', False) and chart_type.lower() not in ['pie', 'pie chart']:
            ax.set_yscale('log')
            
        if enhancements.get('log_scale_x', False) and chart_type.lower() not in ['pie', 'pie chart']:
            ax.set_xscale('log')
            
        # Set grid
        if 'grid' in enhancements:
            ax.grid(enhancements['grid'], linestyle=enhancements.get('grid_style', '--'), alpha=enhancements.get('grid_alpha', 0.7))
            
        # Add annotations
        if 'annotations' in enhancements and isinstance(enhancements['annotations'], list):
            for annotation in enhancements['annotations']:
                if all(k in annotation for k in ['x', 'y', 'text']):
                    ax.annotate(annotation['text'], 
                              xy=(annotation['x'], annotation['y']),
                              xytext=annotation.get('xytext', (annotation['x'], annotation['y'])),
                              arrowprops=annotation.get('arrowprops', dict(arrowstyle='->')),
                              fontsize=annotation.get('fontsize', 10))
        
        # Add legend with custom position
        if 'legend_position' in enhancements and chart_type.lower() not in ['pie', 'pie chart', 'heatmap', 'heat map']:
            ax.legend(loc=enhancements['legend_position'])
        
        # Add text watermark
        if 'watermark' in enhancements:
            fig.text(0.5, 0.01, enhancements['watermark'], 
                   fontsize=enhancements.get('watermark_size', 10),
                   alpha=enhancements.get('watermark_alpha', 0.5),
                   ha='center')
        
        plt.tight_layout()
        
        # Convert the figure to base64
        img_str = _figure_to_base64(fig)
        
        # Generate code for the enhanced chart
        code = """import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
"""
        
        if enhancements.get('add_trendline', False) and chart_type.lower() in ["scatter", "scatter plot"]:
            code += "from scipy import stats\n"
            
        code += f"\n# Create enhanced {chart_type} chart\n"
        code += f"fig, ax = plt.subplots(figsize={enhancements.get('figsize', '(10, 6)')})\n\n"
        
        if 'color_palette' in enhancements:
            code += f"sns.set_palette('{enhancements['color_palette']}')\n\n"
        
        if chart_type.lower() in ["scatter", "scatter plot"]:
            code += f"scatter = ax.scatter(df['{x_column}'], df['{y_column}'], "
            code += f"alpha={enhancements.get('alpha', 0.7)}, "
            code += f"s={enhancements.get('marker_size', 50)})\n"
            
            if enhancements.get('add_trendline', False):
                code += f"\n# Add trendline\n"
                code += f"slope, intercept, r_value, p_value, std_err = stats.linregress(df['{x_column}'], df['{y_column}'])\n"
                code += f"x_line = np.array([df['{x_column}'].min(), df['{x_column}'].max()])\n"
                code += f"y_line = intercept + slope * x_line\n"
                code += f"ax.plot(x_line, y_line, color='red', linestyle='--', label=f'Trend line (r²={r_value**2:.2f})')\n"
                code += f"ax.legend()\n"
        
        elif chart_type.lower() in ["bar", "bar chart"]:
            code += f"bars = sns.barplot("
            code += f"x='{x_column}', y='{y_column}', "
            if 'hue' in enhancements:
                code += f"hue='{enhancements['hue']}', "
            code += f"data=df, ax=ax)\n"
            
            if enhancements.get('add_value_labels', False):
                code += f"\n# Add value labels on top of bars\n"
                code += f"for p in bars.patches:\n"
                code += f"    bars.annotate(f'{{p.get_height():.1f}}', \n"
                code += f"                 (p.get_x() + p.get_width() / 2., p.get_height()),\n"
                code += f"                 ha='center', va='bottom', fontsize=10)\n"
        
        elif chart_type.lower() in ["line", "line chart"]:
            if 'hue' in enhancements:
                code += f"lines = sns.lineplot(x='{x_column}', y='{y_column}', "
                code += f"hue='{enhancements['hue']}', "
                code += f"marker='{enhancements.get('marker', 'o')}', "
                code += f"data=df, ax=ax)\n"
            else:
                code += f"lines = ax.plot(df['{x_column}'], df['{y_column}'], "
                code += f"marker='{enhancements.get('marker', 'o')}')\n"
            
            if enhancements.get('fill_area', False) and 'hue' not in enhancements:
                code += f"\n# Add area fill under the line\n"
                code += f"ax.fill_between(df['{x_column}'], df['{y_column}'], alpha=0.2)\n"
        
        elif chart_type.lower() in ["pie", "pie chart"]:
            if y_column:
                code += f"values = df.groupby('{x_column}')['{y_column}'].sum()\n"
            else:
                code += f"values = df['{x_column}'].value_counts()\n"
            
            code += f"\nwedges, texts, autotexts = ax.pie(\n"
            code += f"    values, \n"
            if not enhancements.get('remove_labels', False):
                code += f"    labels=values.index,\n"
            else:
                code += f"    labels=None,\n"
                
            if enhancements.get('show_percentages', True):
                code += f"    autopct='%1.1f%%',\n"
            else:
                code += f"    autopct=None,\n"
                
            code += f"    shadow={str(enhancements.get('shadow', False))},\n"
            
            if 'explode' in enhancements:
                code += f"    explode=[0.1] * len(values),\n"
                
            code += f"    startangle={enhancements.get('startangle', 90)}\n"
            code += f")\n"
            
            if enhancements.get('show_percentages', True):
                code += f"\n# Customize text appearance\n"
                code += f"for autotext in autotexts:\n"
                code += f"    autotext.set_color('{enhancements.get('percent_text_color', 'white')}')\n"
                code += f"    autotext.set_fontsize({enhancements.get('percent_text_size', 10)})\n"
            
            code += f"\nax.axis('equal')\n"
        
        elif chart_type.lower() in ["histogram"]:
            code += f"hist = ax.hist(df['{x_column}'], "
            code += f"bins={enhancements.get('bins', 10)}, "
            code += f"color='{enhancements.get('color', 'blue')}', "
            code += f"alpha={enhancements.get('alpha', 0.7)}, "
            code += f"edgecolor='{enhancements.get('edge_color', 'black')}')\n"
            
            if enhancements.get('add_kde', False):
                code += f"\n# Add KDE curve\n"
                code += f"from scipy import stats\n"
                code += f"kde_xs = np.linspace(df['{x_column}'].min(), df['{x_column}'].max(), 100)\n"
                code += f"kde = stats.gaussian_kde(df['{x_column}'].dropna())\n"
                code += f"kde_ys = kde(kde_xs)\n\n"
                code += f"# Scale KDE to match histogram height\n"
                code += f"hist_height = hist[0].max()\n"
                code += f"kde_height = kde_ys.max()\n"
                code += f"scaling_factor = hist_height / kde_height\n\n"
                code += f"twin_ax = ax.twinx()\n"
                code += f"twin_ax.plot(kde_xs, kde_ys * scaling_factor, 'r-', linewidth=2)\n"
                code += f"twin_ax.set_yticks([])\n"
        
        elif chart_type.lower() in ["boxplot", "box plot"]:
            if y_column:
                code += f"sns.boxplot("
                code += f"x='{x_column}', y='{y_column}', "
                if 'hue' in enhancements:
                    code += f"hue='{enhancements['hue']}', "
                code += f"data=df, ax=ax"
                if 'width' in enhancements:
                    code += f", width={enhancements['width']}"
                code += ")\n"
                
                if enhancements.get('add_swarmplot', False):
                    code += f"\n# Add swarmplot overlay\n"
                    code += f"sns.swarmplot(x='{x_column}', y='{y_column}', data=df, ax=ax, "
                    code += f"color='{enhancements.get('swarm_color', '.25')}', "
                    code += f"size={enhancements.get('swarm_size', 5)}, "
                    code += f"alpha={enhancements.get('swarm_alpha', 0.5)})\n"
            else:
                code += f"sns.boxplot(y='{x_column}', data=df, ax=ax)\n"
                
                if enhancements.get('add_swarmplot', False):
                    code += f"\n# Add swarmplot overlay\n"
                    code += f"sns.swarmplot(y='{x_column}', data=df, ax=ax, "
                    code += f"color='{enhancements.get('swarm_color', '.25')}', "
                    code += f"size={enhancements.get('swarm_size', 5)}, "
                    code += f"alpha={enhancements.get('swarm_alpha', 0.5)})\n"
        
        elif chart_type.lower() in ["heatmap", "heat map"]:
            if 'value_column' in enhancements:
                value_col = enhancements['value_column']
                code += f"value_col = '{value_col}'\n"
            else:
                code += f"# Find a suitable numeric column for values\n"
                code += f"numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]\n"
                code += f"numeric_cols = [col for col in numeric_cols if col != '{x_column}' and col != '{y_column}']\n"
                code += f"value_col = numeric_cols[0]  # Using first available numeric column\n"
            
            code += f"\npivot_table = df.pivot_table(index='{y_column}', columns='{x_column}', values=value_col)\n\n"
            code += f"sns.heatmap(pivot_table, \n"
            code += f"          annot={str(enhancements.get('show_values', True))}, \n"
            code += f"          cmap='{enhancements.get('colormap', 'viridis')}', \n"
            code += f"          linewidths={enhancements.get('linewidths', 0.5)}, \n"
            code += f"          fmt='{enhancements.get('value_format', '.2f')}', \n"
            code += f"          ax=ax)\n"
        
        # Add common enhancements to code
        code += f"\n# Apply common enhancements\n"
        
        if title:
            code += f"ax.set_title('{title}', "
            code += f"fontsize={enhancements.get('title_fontsize', 14)}, "
            if enhancements.get('bold_title', False):
                code += f"fontweight='bold')\n"
            else:
                code += f"fontweight='normal')\n"
        
        if 'x_label' in enhancements:
            code += f"ax.set_xlabel('{enhancements['x_label']}', fontsize={enhancements.get('label_fontsize', 12)})\n"
        else:
            code += f"ax.set_xlabel('{x_column}', fontsize={enhancements.get('label_fontsize', 12)})\n"
            
        if 'y_label' in enhancements and chart_type.lower() not in ['pie', 'pie chart']:
            code += f"ax.set_ylabel('{enhancements['y_label']}', fontsize={enhancements.get('label_fontsize', 12)})\n"
        elif y_column and chart_type.lower() not in ['pie', 'pie chart']:
            code += f"ax.set_ylabel('{y_column}', fontsize={enhancements.get('label_fontsize', 12)})\n"
            
        if enhancements.get('rotate_xticks', False):
            code += f"plt.xticks(rotation={enhancements.get('xtick_rotation', 45)}, ha='right')\n"
            
        if enhancements.get('log_scale_y', False) and chart_type.lower() not in ['pie', 'pie chart']:
            code += f"ax.set_yscale('log')\n"
            
        if enhancements.get('log_scale_x', False) and chart_type.lower() not in ['pie', 'pie chart']:
            code += f"ax.set_xscale('log')\n"
            
        if 'grid' in enhancements:
            code += f"ax.grid({str(enhancements['grid'])}, "
            code += f"linestyle='{enhancements.get('grid_style', '--')}', "
            code += f"alpha={enhancements.get('grid_alpha', 0.7)})\n"
            
        if 'annotations' in enhancements and isinstance(enhancements['annotations'], list):
            code += f"\n# Add annotations\n"
            for i, annotation in enumerate(enhancements['annotations']):
                if all(k in annotation for k in ['x', 'y', 'text']):
                    code += f"ax.annotate('{annotation['text']}', \n"
                    code += f"          xy=({annotation['x']}, {annotation['y']}),\n"
                    
                    if 'xytext' in annotation:
                        code += f"          xytext=({annotation['xytext'][0]}, {annotation['xytext'][1]}),\n"
                    else:
                        code += f"          xytext=({annotation['x']}, {annotation['y']}),\n"
                        
                    if 'arrowprops' in annotation:
                        code += f"          arrowprops=dict(arrowstyle='->'),\n"
                        
                    code += f"          fontsize={annotation.get('fontsize', 10)})\n"
        
        if 'legend_position' in enhancements and chart_type.lower() not in ['pie', 'pie chart', 'heatmap', 'heat map']:
            code += f"ax.legend(loc='{enhancements['legend_position']}')\n"
        
        if 'watermark' in enhancements:
            code += f"\n# Add watermark\n"
            code += f"fig.text(0.5, 0.01, '{enhancements['watermark']}', \n"
            code += f"       fontsize={enhancements.get('watermark_size', 10)},\n"
            code += f"       alpha={enhancements.get('watermark_alpha', 0.5)},\n"
            code += f"       ha='center')\n"
        
        code += f"\nplt.tight_layout()\n"
        code += f"plt.show()\n"
        
        return {
            "code": code,
            "image": img_str
        }
    
    except Exception as e:
        return {"error": f"Error creating enhanced chart: {str(e)}"}

@tool
def get_sample_data() -> str:
    """
    Return the first 5 rows of the dataset.
    
    Returns:
        A string representation of the first 5 rows of the dataset.
    """
    if dataset.df is None:
        return "No dataset has been loaded. Please load a dataset first."
    
    return dataset.df.head().to_string()

# Set up the LangChain agent
tools = [
    Tool(
        name="load_dataset",
        func=load_dataset,
        description="Load a CSV dataset from a file path"
    ),
    Tool(
        name="analyze_dataset",
        func=analyze_dataset,
        description="Analyze the loaded dataset and return summary information"
    ),
    Tool(
        name="recommend_charts",
        func=recommend_charts,
        description="Recommend appropriate chart types based on the dataset"
    ),
    Tool(
        name="validate_chart",
        func=validate_chart,
        description="Validate if a chart type is appropriate for selected columns"
    ),
    Tool(
        name="generate_chart_code",
        func=generate_chart_code,
        description="Generate Python code to create a specified chart"
    ),
    Tool(
        name="create_chart",
        func=create_chart,
        description="Create and return a chart based on specified parameters"
    ),
    Tool(
        name="enhance_chart",
        func=enhance_chart,
        description="Create an enhanced chart with additional formatting and features"
    ),
    Tool(
        name="get_sample_data",
        func=get_sample_data,
        description="Return the first 5 rows of the dataset"
    )
]

# Initialize the OpenAI model
llm = ChatOllama(model="llama3.2:latest", temperature=0)

# Create the agent
prompt = PromptTemplate.from_template("""You are a helpful data analysis assistant that specializes in data visualization.
You have access to a dataset and can help analyze it and create visualizations.

Follow these steps:
1. First, ask the user to provide a path to their CSV file, unless they already did.
2. Analyze the dataset to understand its structure and content.
3. Recommend appropriate chart types based on the data.
4. Help the user create and refine visualizations.

Always be conversational and explain your recommendations in terms anyone can understand.
When recommending charts, explain why they're appropriate for the data.

{input}
""")

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_data_analysis_agent(user_input):
    """Run the data analysis agent with the given user input."""
    return agent_executor.invoke({"input": user_input})

# Example usage
if __name__ == "__main__":
    print("Welcome to the Data Analysis Agent!")
    print("Please provide a path to your CSV file to begin.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        response = run_data_analysis_agent(user_input)
        print(f"\nAgent: {response['output']}")