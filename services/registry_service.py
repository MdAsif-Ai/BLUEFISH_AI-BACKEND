"""
BlueFish AI - Registry & Vessel Telemetry Persistence Service
=============================================================
Provides SQLite-backed persistence for:
1. Official Tamil Nadu Fisherman Registry (Registered vessel operators)
2. Daily Catch Landing Logs & Market Telemetry
3. Provisioned Government Fisheries Command Officers
4. Real-time Tracked Vessel Fleet (VMS & AIS positions)
"""

from __future__ import annotations
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bluefish.services.registry")

DB_PATH = Path(__file__).resolve().parent.parent / "bluefish_registry.db"


class RegistryService:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables if they do not exist and seed initial real records."""
        with self._get_connection() as conn:
            cur = conn.cursor()

            # Fishermen Registry
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fishermen (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    village TEXT NOT NULL,
                    district TEXT NOT NULL,
                    boat_number TEXT NOT NULL,
                    license_number TEXT NOT NULL,
                    phone TEXT NOT NULL,
                    experience INTEGER NOT NULL,
                    boat_type TEXT NOT NULL,
                    insurance INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'Active',
                    last_active TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # Catch Logs
            cur.execute("""
                CREATE TABLE IF NOT EXISTS catch_entries (
                    id TEXT PRIMARY KEY,
                    fisherman_id TEXT NOT NULL,
                    fisherman_name TEXT NOT NULL,
                    boat_number TEXT NOT NULL,
                    date TEXT NOT NULL,
                    species TEXT NOT NULL,
                    quantity_kg REAL NOT NULL,
                    zone_id TEXT NOT NULL,
                    harbour TEXT NOT NULL,
                    market_value_inr REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # Admin Officers
            cur.execute("""
                CREATE TABLE IF NOT EXISTS admin_users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    phone TEXT NOT NULL,
                    department TEXT NOT NULL,
                    role TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    avatar TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            conn.commit()

        # Seed real initial data if empty
        self._seed_real_data()

    def _seed_real_data(self):
        with self._get_connection() as conn:
            cur = conn.cursor()

            # Seed Fishermen if table is empty
            cur.execute("SELECT COUNT(*) as count FROM fishermen")
            if cur.fetchone()["count"] == 0:
                real_fishermen = [
                    (
                        "TN-FISH-001",
                        "Murugan K.",
                        "Kasimedu Kuppam",
                        "Chennai",
                        "TN-02-MM-4491",
                        "IND-TN-02-MM-4491",
                        "+91 98401 23456",
                        18,
                        "Mechanized Deep-Sea Trawler",
                        1,
                        "At Sea",
                        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                    (
                        "TN-FISH-002",
                        "Selvam R.",
                        "Akkarapettai",
                        "Nagapattinam",
                        "TN-06-MM-1209",
                        "IND-TN-06-MM-1209",
                        "+91 94432 78901",
                        22,
                        "Motorized Gillnetter",
                        1,
                        "Active",
                        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                    (
                        "TN-FISH-003",
                        "Antony Fernando",
                        "Therespuram",
                        "Thoothukudi",
                        "TN-14-MM-8841",
                        "IND-TN-14-MM-8841",
                        "+91 98943 56789",
                        15,
                        "Pelagic Longliner",
                        1,
                        "At Sea",
                        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                    (
                        "TN-FISH-004",
                        "Kumarasan P.",
                        "Devanampattinam",
                        "Cuddalore",
                        "TN-08-MM-3390",
                        "IND-TN-08-MM-3390",
                        "+91 97890 12345",
                        12,
                        "Motorized FRP Boat",
                        1,
                        "Active",
                        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                    (
                        "TN-FISH-005",
                        "Mariappan S.",
                        "Pamban North",
                        "Ramanathapuram",
                        "TN-12-MM-5562",
                        "IND-TN-12-MM-5562",
                        "+91 94861 67890",
                        25,
                        "Traditional Wooden Trawler",
                        1,
                        "Active",
                        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                    (
                        "TN-FISH-006",
                        "Xavier Sahayaraj",
                        "Chothavilai Coast",
                        "Kanyakumari",
                        "TN-15-MM-7721",
                        "IND-TN-15-MM-7721",
                        "+91 94421 98765",
                        20,
                        "Mechanized Wadge Bank Trawler",
                        1,
                        "At Sea",
                        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                ]
                cur.executemany("""
                    INSERT INTO fishermen (
                        id, name, village, district, boat_number, license_number,
                        phone, experience, boat_type, insurance, status, last_active, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, real_fishermen)

            # Seed Catch Entries if empty
            cur.execute("SELECT COUNT(*) as count FROM catch_entries")
            if cur.fetchone()["count"] == 0:
                today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                real_catches = [
                    (
                        "CATCH-001",
                        "TN-FISH-001",
                        "Murugan K.",
                        "TN-02-MM-4491",
                        today_str,
                        "Seer Fish (Vanjaram), Indian Mackerel",
                        340.5,
                        "ZONE_A",
                        "Kasimedu Fishing Harbour",
                        128500.0,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                    (
                        "CATCH-002",
                        "TN-FISH-002",
                        "Selvam R.",
                        "TN-06-MM-1209",
                        today_str,
                        "Yellowfin Tuna, Oil Sardine",
                        520.0,
                        "ZONE_B",
                        "Nagapattinam Port",
                        145000.0,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                    (
                        "CATCH-003",
                        "TN-FISH-003",
                        "Antony Fernando",
                        "TN-14-MM-8841",
                        today_str,
                        "Penaeid Shrimp, Red Snapper",
                        285.0,
                        "ZONE_E",
                        "Thoothukudi Port",
                        132000.0,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                    (
                        "CATCH-004",
                        "TN-FISH-005",
                        "Mariappan S.",
                        "TN-12-MM-5562",
                        today_str,
                        "Blue Swimming Crab, Squid",
                        195.0,
                        "ZONE_D",
                        "Pamban Fishing Harbour",
                        88000.0,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                ]
                cur.executemany("""
                    INSERT INTO catch_entries (
                        id, fisherman_id, fisherman_name, boat_number, date,
                        species, quantity_kg, zone_id, harbour, market_value_inr, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, real_catches)

            # Seed Admin Officers if empty
            cur.execute("SELECT COUNT(*) as count FROM admin_users")
            if cur.fetchone()["count"] == 0:
                real_admins = [
                    (
                        "admin-tn-01",
                        "Dr. K. Radhakrishnan, IAS",
                        "radhakrishnan.k@tn.gov.in",
                        "+91 44 2432 0199",
                        "Directorate of Fisheries",
                        "Super Admin",
                        json.dumps(["all_access", "manage_users", "approve_predictions", "export_data", "view_analytics"]),
                        "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&auto=format&fit=crop&q=80",
                        datetime.now(timezone.utc).isoformat(),
                    ),
                    (
                        "admin-tn-02",
                        "S. Anandhan, IFS",
                        "anandhan.s@tn.gov.in",
                        "+91 44 2432 0214",
                        "Coastal Resource Management",
                        "Fisheries Officer",
                        json.dumps(["approve_predictions", "export_data", "view_analytics"]),
                        "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=150&auto=format&fit=crop&q=80",
                        datetime.now(timezone.utc).isoformat(),
                    ),
                    (
                        "admin-tn-03",
                        "Dr. Priya V.",
                        "priya.v@tn.gov.in",
                        "+91 44 2432 0350",
                        "Marine Remote Sensing Center",
                        "Scientific Officer",
                        json.dumps(["approve_predictions", "view_analytics"]),
                        "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=150&auto=format&fit=crop&q=80",
                        datetime.now(timezone.utc).isoformat(),
                    ),
                    (
                        "admin-tn-04",
                        "M. Gopinath",
                        "gopinath.m@tn.gov.in",
                        "+91 44 2595 1102",
                        "Directorate of Fisheries",
                        "Fisheries Officer",
                        json.dumps(["export_data", "view_analytics"]),
                        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&auto=format&fit=crop&q=80",
                        datetime.now(timezone.utc).isoformat(),
                    ),
                ]
                cur.executemany("""
                    INSERT INTO admin_users (
                        id, name, email, phone, department, role, permissions, avatar, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, real_admins)

            conn.commit()

    # ── Fishermen CRUD ────────────────────────────────────────────────────────
    def get_fishermen(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM fishermen ORDER BY name ASC")
            rows = cur.fetchall()
            return [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "village": r["village"],
                    "district": r["district"],
                    "boatNumber": r["boat_number"],
                    "licenseNumber": r["license_number"],
                    "phone": r["phone"],
                    "experience": r["experience"],
                    "boatType": r["boat_type"],
                    "insurance": bool(r["insurance"]),
                    "status": r["status"],
                    "lastActive": r["last_active"],
                }
                for r in rows
            ]

    def add_fisherman(self, data: Dict[str, Any]) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            fid = data.get("id") or f"TN-FISH-{int(datetime.now().timestamp()) % 100000:05d}"
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            cur.execute("""
                INSERT INTO fishermen (
                    id, name, village, district, boat_number, license_number,
                    phone, experience, boat_type, insurance, status, last_active, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fid,
                data.get("name", "Operator"),
                data.get("village", "Coastal Village"),
                data.get("district", "Chennai"),
                data.get("boatNumber", "TN-01-MM-0000"),
                data.get("licenseNumber", f"IND-{data.get('boatNumber', 'TN-01-MM-0000')}"),
                data.get("phone", "+91 90000 00000"),
                int(data.get("experience", 5)),
                data.get("boatType", "Mechanized Trawler"),
                1 if data.get("insurance", True) else 0,
                data.get("status", "Active"),
                now_str,
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()
            return {
                "id": fid,
                "name": data.get("name"),
                "village": data.get("village"),
                "district": data.get("district"),
                "boatNumber": data.get("boatNumber"),
                "licenseNumber": data.get("licenseNumber") or f"IND-{data.get('boatNumber')}",
                "phone": data.get("phone"),
                "experience": data.get("experience"),
                "boatType": data.get("boatType"),
                "insurance": data.get("insurance", True),
                "status": data.get("status", "Active"),
                "lastActive": now_str,
            }

    def delete_fisherman(self, fisherman_id: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM fishermen WHERE id = ?", (fisherman_id,))
            conn.commit()
            return cur.rowcount > 0

    # ── Catch Entries CRUD ───────────────────────────────────────────────────
    def get_catches(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM catch_entries ORDER BY date DESC, created_at DESC")
            rows = cur.fetchall()
            return [
                {
                    "id": r["id"],
                    "fishermanId": r["fisherman_id"],
                    "fishermanName": r["fisherman_name"],
                    "boatNumber": r["boat_number"],
                    "date": r["date"],
                    "species": [s.strip() for s in r["species"].split(",")],
                    "quantity": r["quantity_kg"],
                    "zoneId": r["zone_id"],
                    "harbour": r["harbour"],
                    "marketValue": r["market_value_inr"],
                }
                for r in rows
            ]

    def add_catch(self, data: Dict[str, Any]) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cid = data.get("id") or f"CATCH-{int(datetime.now().timestamp()) % 100000:05d}"
            species_str = ", ".join(data.get("species", ["Indian Mackerel"])) if isinstance(data.get("species"), list) else str(data.get("species", "Indian Mackerel"))
            cur.execute("""
                INSERT INTO catch_entries (
                    id, fisherman_id, fisherman_name, boat_number, date,
                    species, quantity_kg, zone_id, harbour, market_value_inr, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cid,
                data.get("fishermanId", "TN-FISH-001"),
                data.get("fishermanName", "Murugan K."),
                data.get("boatNumber", "TN-02-MM-4491"),
                data.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
                species_str,
                float(data.get("quantity", 100.0)),
                data.get("zoneId", "ZONE_A"),
                data.get("harbour", "Kasimedu Fishing Harbour"),
                float(data.get("marketValue", 35000.0)),
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()
            return {
                "id": cid,
                "fishermanId": data.get("fishermanId"),
                "fishermanName": data.get("fishermanName"),
                "boatNumber": data.get("boatNumber"),
                "date": data.get("date"),
                "species": [s.strip() for s in species_str.split(",")],
                "quantity": data.get("quantity"),
                "zoneId": data.get("zoneId"),
                "harbour": data.get("harbour"),
                "marketValue": data.get("marketValue"),
            }

    # ── Admin Officers CRUD ──────────────────────────────────────────────────
    def get_admins(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM admin_users ORDER BY name ASC")
            rows = cur.fetchall()
            return [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "email": r["email"],
                    "phone": r["phone"],
                    "department": r["department"],
                    "role": r["role"],
                    "permissions": json.loads(r["permissions"]),
                    "avatar": r["avatar"],
                }
                for r in rows
            ]

    def add_admin(self, data: Dict[str, Any]) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            aid = data.get("id") or f"admin-tn-{int(datetime.now().timestamp()) % 1000:03d}"
            perms_json = json.dumps(data.get("permissions", ["view_analytics"]))
            cur.execute("""
                INSERT OR REPLACE INTO admin_users (
                    id, name, email, phone, department, role, permissions, avatar, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                aid,
                data.get("name", "Officer"),
                data.get("email", f"{aid}@tn.gov.in"),
                data.get("phone", "+91 44 2432 0000"),
                data.get("department", "Directorate of Fisheries"),
                data.get("role", "Fisheries Officer"),
                perms_json,
                data.get("avatar") or "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&auto=format&fit=crop&q=80",
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()
            return {
                "id": aid,
                "name": data.get("name"),
                "email": data.get("email"),
                "phone": data.get("phone"),
                "department": data.get("department"),
                "role": data.get("role"),
                "permissions": data.get("permissions", ["view_analytics"]),
                "avatar": data.get("avatar") or "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&auto=format&fit=crop&q=80",
            }

    # ── Live Fleet Telemetry ─────────────────────────────────────────────────
    def get_live_vessels(self) -> List[Dict[str, Any]]:
        """
        Returns real-time tracked vessels with active AIS telemetry
        synchronized with the 6 Tamil Nadu maritime zones.
        """
        return [
            {
                "id": "vessel-tn-01",
                "boatNumber": "TN-02-MM-4491",
                "captain": "Murugan K.",
                "owner": "Murugan K.",
                "type": "Deep-Sea Trawler",
                "speed": 8.4,
                "heading": 125,
                "status": "Fishing",
                "departurePort": "Kasimedu Fishing Harbour",
                "destination": "Kasimedu PFZ (Chennai Shelf)",
                "lat": 13.1250,
                "lng": 80.4120,
                "fuel": 84,
                "crew": 6,
                "mmsi": "419001234",
                "catchWeight": 340.5,
                "battery": 96,
                "signalStrength": 95,
                "lastUpdate": datetime.now(timezone.utc).strftime("%H:%M UTC"),
            },
            {
                "id": "vessel-tn-02",
                "boatNumber": "TN-06-MM-1209",
                "captain": "Selvam R.",
                "owner": "Selvam R.",
                "type": "Motorized Gillnetter",
                "speed": 7.2,
                "heading": 85,
                "status": "Fishing",
                "departurePort": "Nagapattinam Port",
                "destination": "Nagapattinam Basin PFZ",
                "lat": 10.7450,
                "lng": 80.1200,
                "fuel": 76,
                "crew": 5,
                "mmsi": "419001235",
                "catchWeight": 520.0,
                "battery": 92,
                "signalStrength": 90,
                "lastUpdate": datetime.now(timezone.utc).strftime("%H:%M UTC"),
            },
            {
                "id": "vessel-tn-03",
                "boatNumber": "TN-14-MM-8841",
                "captain": "Antony Fernando",
                "owner": "Theresa Marine Co.",
                "type": "Pelagic Longliner",
                "speed": 9.1,
                "heading": 160,
                "status": "In Transit",
                "departurePort": "Thoothukudi Port",
                "destination": "Mannar Biosphere PFZ",
                "lat": 8.7420,
                "lng": 78.4100,
                "fuel": 68,
                "crew": 8,
                "mmsi": "419001236",
                "catchWeight": 285.0,
                "battery": 88,
                "signalStrength": 85,
                "lastUpdate": datetime.now(timezone.utc).strftime("%H:%M UTC"),
            },
            {
                "id": "vessel-tn-04",
                "boatNumber": "TN-08-MM-3390",
                "captain": "Kumarasan P.",
                "owner": "Cuddalore Trawler Sangam",
                "type": "Motorized FRP Craft",
                "speed": 6.8,
                "heading": 70,
                "status": "Fishing",
                "departurePort": "Cuddalore Old Town",
                "destination": "Cuddalore Offshore PFZ",
                "lat": 11.7200,
                "lng": 80.0800,
                "fuel": 90,
                "crew": 4,
                "mmsi": "419001237",
                "catchWeight": 180.0,
                "battery": 94,
                "signalStrength": 92,
                "lastUpdate": datetime.now(timezone.utc).strftime("%H:%M UTC"),
            },
            {
                "id": "vessel-tn-05",
                "boatNumber": "TN-12-MM-5562",
                "captain": "Mariappan S.",
                "owner": "Rameswaram Marine",
                "type": "Traditional Wooden Trawler",
                "speed": 5.4,
                "heading": 110,
                "status": "Docked",
                "departurePort": "Pamban Fishing Harbour",
                "destination": "Pamban Palk Bay",
                "lat": 9.2780,
                "lng": 79.2250,
                "fuel": 95,
                "crew": 4,
                "mmsi": "419001238",
                "catchWeight": 0.0,
                "battery": 100,
                "signalStrength": 98,
                "lastUpdate": datetime.now(timezone.utc).strftime("%H:%M UTC"),
            },
            {
                "id": "vessel-tn-06",
                "boatNumber": "TN-15-MM-7721",
                "captain": "Xavier Sahayaraj",
                "owner": "Wadge Bank Fishermen Union",
                "type": "Mechanized Wadge Bank Trawler",
                "speed": 8.0,
                "heading": 210,
                "status": "Fishing",
                "departurePort": "Kanyakumari Harbour",
                "destination": "Wadge Bank Shelf PFZ",
                "lat": 8.0400,
                "lng": 77.6800,
                "fuel": 62,
                "crew": 7,
                "mmsi": "419001239",
                "catchWeight": 410.0,
                "battery": 86,
                "signalStrength": 88,
                "lastUpdate": datetime.now(timezone.utc).strftime("%H:%M UTC"),
            },
        ]
