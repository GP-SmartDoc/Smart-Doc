from graphs.visualization_graph import generate_visualization
from states.visualization_state import DiagramType
# answer = generate_visualization(
#     description="""generate a state diagram that describes a video game player.
#         He is either active, tired, energized, or dead. He is active by default. If he gets 
#         an energy drink while tired nothing happens. if he gets it while active he becomes energized.
#         If he gets it while energized nothing happens. If he gets hit while tired he dies, if he gets
#         hit while active he becomes tired, and if he gets hit while energized he becomes active"
#     """,
#     type=DiagramType.STATE
# )
# print(answer)

# answer = generate_visualization(
#     description="""
#         generate a flowchart about for the gcd algorithm
#     """,
#     type=DiagramType.FLOWCHART
# )

answer = generate_visualization(
    description= """
        draw a class diagram that represent 
    """,
    type=DiagramType.CLASS
)