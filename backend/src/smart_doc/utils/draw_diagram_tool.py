from langchain_core.tools import tool

@tool
def draw_diagrams(descriptions:str) -> str:
    """Creates diagrams based on description. Any number of diagrams"""

    '''
    description of diagrams should be in this form :
    {  
        {
            "type" : "...",
            "description" : "..."
        },
        {
            "type" : "...",
            "description" : "..."
        }
    }
    '''

    '''
        mermaid_diagrams = []
        for each type, description in descriptions:
            prompt = retireve_prompt_for_type(type)
            output = visuualization_graph.invoke()
            mermaid_diagrams.append(output)    
            
        return 
    '''

tools = [draw_diagrams]
