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
    Generate an ERD using Crow's Foot notation for a 'Global Airline Fleet & Flight Management System'. The diagram must include accurate primary keys (PK), foreign keys (FK), and data types.

    System Requirements:

        Airports: Store the 3-letter IATA airport_code (PK), name, and city.

        Routes: A route connects two airports. It needs an origin_code (FK) and destination_code (FK). Constraint: An airport can be the origin for many routes and the destination for many routes, but a route must have exactly one origin and one destination.

        Flight Instances (Weak Entity): A Route is just a plan. A FlightInstance represents a specific plane flying a route on a specific date. It is a weak entity dependent on the Route. Its primary key must be a composite key made of the route_id (FK) and the departure_date (TIMESTAMP).

        Aircraft: Store tail_number (PK), model, and capacity. A FlightInstance must be assigned exactly one Aircraft, but an Aircraft can be assigned to zero-to-many FlightInstances over time.

        Employees (Self-Referencing): Store employee_id (PK, UUID), name, and role. Include a recursive relationship where an Employee can manage zero-to-many other Employees (e.g., a supervisor_id acting as an FK to employee_id).

        Crew Assignment (Many-to-Many): Employees are assigned to FlightInstances. Because many employees work on many flights, resolve this many-to-many relationship using an associative/junction table called CrewRoster. This table must include an additional attribute: shift_role (VARCHAR).

        Bookings & Tickets (One-to-One): A Passenger makes a Booking (PK: booking_reference). Each successful Booking generates exactly one Ticket (PK: ticket_number). A Ticket cannot exist without a Booking.
    """,
    type=DiagramType.ER
)
print(answer)
# answer = generate_visualization(
#     description= """
#     Organic Search (Google/Bing): 1,450,200

#     Direct Traffic: 980,500

#     Paid Search (Google Ads): 725,000

#     Social (Instagram/Facebook): 510,100

#     Social (TikTok): 480,900

#     Email Newsletters: 215,400

#     Affiliate Referrals: 150,200

#     Display Banners: 95,800

#     Social (LinkedIn): 12,500

#     Offline (QR Codes): 3,200

#     SMS Links: 850
#     """,
#     type=DiagramType.PIE
# )
