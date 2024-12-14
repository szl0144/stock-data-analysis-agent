
from IPython.display import Image


def get_mermaid_flowchart(app):
    """
    Get the mermaid flowchart from the app
    
    Parameters
    ----------
    app : LangGraph
        The app to get the mermaid flowchart from
    
    Returns
    -------
    Image
        The mermaid flowchart
    
    """
    return Image(app.get_graph().draw_mermaid_png())