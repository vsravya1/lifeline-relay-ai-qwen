"""
Resource inventory for relief allocation.

This is intentionally a simple, editable starting point — in a real
deployment this would sync from an actual logistics system, but for
the hackathon demo it's a fixed inventory that creates genuine scarcity:
not every zone can get a helicopter, so the allocation has to make
real tradeoffs, not just hand out unlimited resources everywhere.

Vehicles and materials are tracked separately and depleted independently
as zones get approved allocations.
"""

# Vehicles: each unit can be assigned to exactly one zone.
INITIAL_VEHICLE_INVENTORY = {
    "helicopter": 1,    # fastest, but extremely limited — reserved for the most severe/inaccessible zone
    "boat": 3,          # the workhorse for flood response
    "ground_vehicle": 2,  # for zones with shallower water, road access still viable
}

# Relief materials: counted in generic "supply units" per type, can be
# split across multiple zones (unlike vehicles, which are indivisible).
INITIAL_MATERIAL_INVENTORY = {
    "food": 10,
    "medicine": 6,
    "life_jackets": 8,
    "clean_water": 10,
}
