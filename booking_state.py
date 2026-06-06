"""
booking_state.py — Session-level booking state tracker for the Himachal Homestay Agent.

This is a NEW module (no equivalent in the Feynman pipeline).
It tracks the in-progress booking details across conversation turns
and manages the booking confirmation lifecycle.

State lives in memory during the session; on confirmation it is
persisted to a local JSON bookings log (bookings/confirmed_bookings.json).
"""

import os
import json
import random
import string
from datetime import datetime


BOOKINGS_FILE = "./bookings/confirmed_bookings.json"

# Booking lifecycle stages
STAGE_IDLE = "idle"                  # No booking in progress
STAGE_COLLECTING = "collecting"      # Gathering guest details
STAGE_CONFIRMING = "confirming"      # Summary shown, awaiting guest confirmation
STAGE_CONFIRMED = "confirmed"        # Booking confirmed, reference ID issued


def _generate_booking_ref() -> str:
    """Generates a booking reference in format HP-YYYYMMDD-XXXX."""
    date_str = datetime.now().strftime("%Y%m%d")
    suffix = ''.join(random.choices(string.digits, k=4))
    return f"HP-{date_str}-{suffix}"


def empty_booking() -> dict:
    """Returns a blank booking state dict."""
    return {
        "stage": STAGE_IDLE,
        "ref_id": None,

        # Guest details
        "guest_name": None,
        "guest_contact": None,       # WhatsApp / email

        # Trip details
        "destination": None,
        "check_in": None,
        "check_out": None,
        "num_guests": None,
        "guest_composition": None,   # solo / couple / family / group

        # Room preference
        "room_type": None,           # standard / deluxe / dormitory / cottage
        "special_requirements": None,

        # Itinerary
        "itinerary_title": None,

        # Financial
        "quoted_price": None,        # Price shown to guest

        # Timestamps
        "created_at": None,
        "confirmed_at": None,
    }


class BookingStateManager:
    """
    Manages the booking state for a single conversation session.
    
    Usage:
        bsm = BookingStateManager()
        bsm.update({"destination": "Spiti Valley", "num_guests": 2})
        bsm.advance_stage(STAGE_CONFIRMING)
        summary = bsm.get_summary_text()
        ref = bsm.confirm()
    """

    def __init__(self):
        self.state = empty_booking()

    def update(self, fields: dict):
        """
        Updates booking state fields.
        Only updates fields that are provided (partial updates are safe).
        Automatically moves to COLLECTING stage on first update.
        """
        for k, v in fields.items():
            if k in self.state and v is not None:
                self.state[k] = v

        if self.state["stage"] == STAGE_IDLE and self._has_any_field():
            self.state["stage"] = STAGE_COLLECTING
            self.state["created_at"] = datetime.now().isoformat()

    def _has_any_field(self) -> bool:
        """Returns True if at least one meaningful booking field is set."""
        fields = [
            "guest_name", "destination", "check_in", "check_out",
            "num_guests", "room_type", "itinerary_title"
        ]
        return any(self.state.get(f) for f in fields)

    def missing_fields(self) -> list[str]:
        """
        Returns a list of required fields that are still missing.
        Required fields for confirmation: guest_name, destination,
        check_in, check_out, num_guests, guest_contact.
        """
        required = {
            "guest_name": "your name",
            "destination": "destination / property",
            "check_in": "check-in date",
            "check_out": "check-out date",
            "num_guests": "number of guests",
            "guest_contact": "your WhatsApp number or email"
        }
        return [label for field, label in required.items() if not self.state.get(field)]

    def is_ready_to_confirm(self) -> bool:
        """Returns True when all required fields are collected."""
        return len(self.missing_fields()) == 0

    def advance_stage(self, new_stage: str):
        """Advances the booking to a new lifecycle stage."""
        self.state["stage"] = new_stage

    def confirm(self) -> str:
        """
        Finalizes the booking:
        - Generates a reference ID
        - Records confirmation timestamp
        - Persists to the bookings log
        Returns the booking reference ID.
        """
        ref_id = _generate_booking_ref()
        self.state["ref_id"] = ref_id
        self.state["stage"] = STAGE_CONFIRMED
        self.state["confirmed_at"] = datetime.now().isoformat()

        # Persist to local log
        self._persist()

        return ref_id

    def _persist(self):
        """Appends the confirmed booking to the local JSON bookings log."""
        os.makedirs(os.path.dirname(BOOKINGS_FILE), exist_ok=True)

        bookings = []
        if os.path.exists(BOOKINGS_FILE):
            try:
                with open(BOOKINGS_FILE, 'r', encoding='utf-8') as f:
                    bookings = json.load(f)
            except Exception:
                bookings = []

        bookings.append(self.state.copy())

        with open(BOOKINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(bookings, f, indent=2, ensure_ascii=False)

    def get_summary_text(self) -> str:
        """
        Returns a human-readable booking summary to show the guest
        before they confirm.
        """
        s = self.state
        lines = ["📋 *Booking Summary*"]

        if s["guest_name"]:
            lines.append(f"• Guest: {s['guest_name']}")
        if s["destination"]:
            lines.append(f"• Destination: {s['destination']}")
        if s["itinerary_title"]:
            lines.append(f"• Itinerary: {s['itinerary_title']}")
        if s["room_type"]:
            lines.append(f"• Room Type: {s['room_type']}")
        if s["check_in"]:
            lines.append(f"• Check-in: {s['check_in']}")
        if s["check_out"]:
            lines.append(f"• Check-out: {s['check_out']}")
        if s["num_guests"]:
            lines.append(f"• Number of Guests: {s['num_guests']}")
        if s["guest_composition"]:
            lines.append(f"• Group Type: {s['guest_composition']}")
        if s["quoted_price"]:
            lines.append(f"• Quoted Price: {s['quoted_price']}")
        if s["special_requirements"]:
            lines.append(f"• Special Requirements: {s['special_requirements']}")
        if s["guest_contact"]:
            lines.append(f"• Contact: {s['guest_contact']}")

        return "\n".join(lines)

    def to_prompt_string(self) -> str:
        """
        Returns a concise booking state string suitable for injection
        into the system prompt.
        """
        s = self.state
        if s["stage"] == STAGE_IDLE:
            return "No booking in progress."

        if s["stage"] == STAGE_CONFIRMED:
            return (
                f"Booking CONFIRMED. Reference ID: {s['ref_id']}. "
                f"Guest: {s['guest_name']}, Destination: {s['destination']}, "
                f"Dates: {s['check_in']} to {s['check_out']}, "
                f"Guests: {s['num_guests']}."
            )

        parts = [f"Booking stage: {s['stage'].upper()}."]
        if s["guest_name"]:
            parts.append(f"Guest: {s['guest_name']}.")
        if s["destination"]:
            parts.append(f"Destination: {s['destination']}.")
        if s["check_in"] and s["check_out"]:
            parts.append(f"Dates: {s['check_in']} → {s['check_out']}.")
        if s["num_guests"]:
            parts.append(f"Guests: {s['num_guests']}.")
        if s["room_type"]:
            parts.append(f"Room: {s['room_type']}.")

        missing = self.missing_fields()
        if missing:
            parts.append(f"Still needed: {', '.join(missing)}.")

        return " ".join(parts)

    def reset(self):
        """Resets booking state (guest wants to start a new booking)."""
        self.state = empty_booking()
