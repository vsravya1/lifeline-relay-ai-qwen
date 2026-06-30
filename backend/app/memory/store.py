"""
Disaster Memory — the shared state store that connects all 3 phases.

This is intentionally a simple in-memory store for the hackathon build.
WatcherAgent writes zone risk here. ResponderAgent reads it (to boost
SOS urgency) and writes SOS counts back. CoordinatorAgent reads both
to detect conflicts. RecoveryAgent reads zone history to weight damage
reports. The Decision Timeline reads everything to render the live feed.

For production this would be Redis or a real DB — for the hackathon,
a process-lifetime dict is the right scope: simple, fast, zero infra
dependency, and perfectly demoable.
"""

from typing import Dict, List, Optional
from app.models.schemas import ZoneMemory, TimelineEntry, RiskLevel, ConflictEvent, DamageReport, ResourceAllocation
from app.data.resource_inventory import INITIAL_VEHICLE_INVENTORY, INITIAL_MATERIAL_INVENTORY


class DisasterMemory:
    def __init__(self):
        self.zones: Dict[str, ZoneMemory] = {}
        self.timeline: List[TimelineEntry] = []
        self.conflicts: Dict[str, ConflictEvent] = {}
        self.damage_reports: List[DamageReport] = []
        self.resource_allocations: List[ResourceAllocation] = []
        # Live inventory counts — deducted only when a human APPROVES
        # an allocation, never on the initial AI proposal.
        self.vehicle_inventory: Dict[str, int] = dict(INITIAL_VEHICLE_INVENTORY)
        self.material_inventory: Dict[str, int] = dict(INITIAL_MATERIAL_INVENTORY)
        self._entry_counter = 0
        # Tracks each agent's current display status for the dashboard's
        # agent panel — "idle", "processing", or "conflict_found".
        self.agent_status: Dict[str, str] = {
            "WatcherAgent": "idle",
            "ResponderAgent": "idle",
            "CoordinatorAgent": "idle",
            "RecoveryAgent": "idle",
        }

    def set_agent_status(self, agent_name: str, status: str):
        self.agent_status[agent_name] = status

    def get_agent_status(self) -> Dict[str, str]:
        return self.agent_status

    # --- Zone memory ---

    def get_zone(self, zone_id: str) -> ZoneMemory:
        """Get a zone's memory, creating it with defaults if it doesn't exist yet."""
        if zone_id not in self.zones:
            self.zones[zone_id] = ZoneMemory(zone_id=zone_id)
        return self.zones[zone_id]

    def update_zone_risk(self, zone_id: str, risk_level: RiskLevel):
        zone = self.get_zone(zone_id)
        zone.current_risk_level = risk_level

    def record_sos(self, zone_id: str, is_critical: bool, is_vulnerable: bool):
        zone = self.get_zone(zone_id)
        zone.sos_count += 1
        if is_critical:
            zone.critical_sos_count += 1
        if is_vulnerable:
            zone.vulnerability_flags_count += 1

    def set_active_conflict(self, zone_id: str, conflict_id: Optional[str]):
        zone = self.get_zone(zone_id)
        zone.active_conflict = conflict_id

    # --- Conflicts ---

    def add_conflict(self, conflict: ConflictEvent):
        self.conflicts[conflict.conflict_id] = conflict
        self.set_active_conflict(conflict.zone_id, conflict.conflict_id)

    # --- Damage reports ---

    def add_damage_report(self, report: DamageReport):
        # Keep only the latest report per zone, so the gallery shows
        # one current photo+finding per zone rather than accumulating.
        self.damage_reports = [r for r in self.damage_reports if r.zone_id != report.zone_id]
        self.damage_reports.append(report)

    def get_damage_reports(self) -> List[DamageReport]:
        return sorted(self.damage_reports, key=lambda r: r.zone_id)

    # --- Resource allocations ---

    def add_resource_allocation(self, allocation: ResourceAllocation):
        # Keep only the latest allocation per zone, same pattern as damage reports
        self.resource_allocations = [a for a in self.resource_allocations if a.zone_id != allocation.zone_id]
        self.resource_allocations.append(allocation)

    def get_resource_allocations(self) -> List[ResourceAllocation]:
        return sorted(self.resource_allocations, key=lambda a: a.zone_id)

    def get_inventory(self) -> dict:
        return {
            "vehicles": self.vehicle_inventory,
            "materials": self.material_inventory,
        }

    def deduct_vehicle(self, vehicle_type: str) -> bool:
        """Returns False if there's none left — caller must handle gracefully."""
        if vehicle_type and self.vehicle_inventory.get(vehicle_type, 0) > 0:
            self.vehicle_inventory[vehicle_type] -= 1
            return True
        return False

    def restore_vehicle(self, vehicle_type: str):
        """Used when an approved allocation is later changed/re-approved with a different vehicle."""
        if vehicle_type and vehicle_type in self.vehicle_inventory:
            self.vehicle_inventory[vehicle_type] += 1

    def deduct_materials(self, materials: Dict[str, int]) -> bool:
        """All-or-nothing: only deducts if every requested material has enough stock."""
        for material, qty in materials.items():
            if self.material_inventory.get(material, 0) < qty:
                return False
        for material, qty in materials.items():
            self.material_inventory[material] -= qty
        return True

    def restore_materials(self, materials: Dict[str, int]):
        for material, qty in materials.items():
            if material in self.material_inventory:
                self.material_inventory[material] += qty

    # --- Decision Timeline ---

    def log(self, entry: TimelineEntry):
        self._entry_counter += 1
        self.timeline.append(entry)

    def get_timeline(self, limit: int = 50) -> List[TimelineEntry]:
        """Most recent entries first — what the dashboard renders."""
        return sorted(self.timeline, key=lambda e: e.timestamp, reverse=True)[:limit]

    # --- Dashboard summary ---

    def get_all_zones(self) -> List[ZoneMemory]:
        return list(self.zones.values())


# Single shared instance for the whole app's lifetime — every agent
# imports and uses this same object, which is what makes the phases
# "remember" each other instead of operating in isolation.
disaster_memory = DisasterMemory()
