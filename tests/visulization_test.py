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

# answer = generate_visualization(
#     description= """
#         Design a class diagram for a Smart Home IoT system. The system is built around a House, which is composed of multiple Rooms (lifecycle dependency: if the House is deleted, the Rooms are removed). Each Room aggregates a collection of SmartDevices, but these devices are standalone and can be moved between rooms (weak ownership).

# All devices inherit from an abstract class called SmartDevice, which contains a serialNumber and a status. The status field relies on an enumeration called DeviceState (values: ON, OFF, OFFLINE, ERROR).

# There are two concrete device types: SmartBulb (which has brightness and colorHex) and SecurityCamera (which has resolution).
# To handle connectivity, there is an interface called Networkable with a method connectToWiFi(). The SecurityCamera implements this interface directly, but the SmartBulb does not. Instead, the SmartBulb communicates via a Hub. The Hub also inherits from SmartDevice and implements Networkable, acting as a bridge
#     """,
#     type=DiagramType.CLASS
# )

answer = generate_visualization(
    description= """
    Organic Search (Google/Bing): 1,450,200

    Direct Traffic: 980,500

    Paid Search (Google Ads): 725,000

    Social (Instagram/Facebook): 510,100

    Social (TikTok): 480,900

    Email Newsletters: 215,400

    Affiliate Referrals: 150,200

    Display Banners: 95,800

    Social (LinkedIn): 12,500

    Offline (QR Codes): 3,200

    SMS Links: 850
    """,
    type=DiagramType.PIE
)
