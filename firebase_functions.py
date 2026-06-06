# Firebase Admin SDK functions for VaaniStay Python backend
# Called after every Exotel call to write booking data to Firestore

import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────────────
# 1. Firebase Admin SDK Setup
# ──────────────────────────────────────────────────────────
# Place your firebase_service_account.json in the same directory as main.py
# Download from: Firebase Console → Project Settings → Service Accounts → Generate New Private Key

try:
    cred = credentials.Certificate("vaani-stay-firebase-adminsdk-fbsvc-0dab96fa47.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("🔥 Firebase Admin SDK initialized successfully")
except Exception as e:
    print(f"⚠️ Firebase not initialized: {e}")
    print("   Place firebase_service_account.json in project root to enable Firestore writes")
    db = None


# ──────────────────────────────────────────────────────────
# 2. Write Booking to Firestore
# ──────────────────────────────────────────────────────────
def write_booking_to_firestore(booking: dict) -> str | None:
    """
    Creates a new document in the 'bookings' collection AND
    a corresponding document in the 'notifications' collection.
    
    Args:
        booking: dict with keys:
            - guestName (str)
            - guestPhone (str)
            - checkIn (str "DD/MM/YYYY")
            - checkOut (str "DD/MM/YYYY")
            - roomType (str)
            - numGuests (int)
            - pricePerNight (int)
            - totalPrice (int)
            - specialRequests (str)
            - status ("confirmed" | "enquiry" | "cancelled")
            - transcript (str)
            - callDuration (int, seconds)
    
    Returns:
        New booking document ID, or None if Firebase not initialized
    """
    if db is None:
        print("⚠️ Firebase not initialized, skipping Firestore write")
        return None

    try:
        # Parse date strings to Firestore Timestamps
        check_in = datetime.strptime(booking.get("checkIn", "01/01/2025"), "%d/%m/%Y")
        check_out = datetime.strptime(booking.get("checkOut", "02/01/2025"), "%d/%m/%Y")

        # Create booking document
        booking_data = {
            "guestName": booking.get("guestName", "Unknown Guest"),
            "guestPhone": booking.get("guestPhone", ""),
            "checkIn": check_in,
            "checkOut": check_out,
            "roomType": booking.get("roomType", "Standard"),
            "numGuests": booking.get("numGuests", 1),
            "pricePerNight": booking.get("pricePerNight", 0),
            "totalPrice": booking.get("totalPrice", 0),
            "specialRequests": booking.get("specialRequests", ""),
            "status": booking.get("status", "confirmed"),
            "transcript": booking.get("transcript", ""),
            "callDuration": booking.get("callDuration", 0),
            "createdAt": firestore.SERVER_TIMESTAMP,
        }

        # Write to bookings collection
        _, booking_ref = db.collection("bookings").add(booking_data)
        booking_id = booking_ref.id
        print(f"✅ Booking written to Firestore: {booking_id}")

        # Write notification
        guest_name = booking.get("guestName", "Guest")
        room_type = booking.get("roomType", "Room")
        status = booking.get("status", "confirmed")
        
        notif_data = {
            "title": "New Booking Confirmed" if status == "confirmed" else "New Enquiry Received",
            "body": f"{guest_name} booked a {room_type} for {booking.get('checkIn', '')} - {booking.get('checkOut', '')}",
            "type": "new_booking" if status == "confirmed" else "enquiry",
            "bookingId": booking_id,
            "isRead": False,
            "createdAt": firestore.SERVER_TIMESTAMP,
        }
        db.collection("notifications").add(notif_data)
        print(f"🔔 Notification written for booking: {booking_id}")

        return booking_id

    except Exception as e:
        print(f"❌ Error writing to Firestore: {e}")
        return None


# ──────────────────────────────────────────────────────────
# 3. Update Room in Firestore
# ──────────────────────────────────────────────────────────
def update_room_in_firestore(room_type: str, checkin: str, checkout: str):
    """
    Queries 'rooms' collection for matching roomType,
    then appends blocked dates using Firestore arrayUnion.
    
    Args:
        room_type: e.g., "Deluxe"
        checkin: "DD/MM/YYYY"
        checkout: "DD/MM/YYYY"
    """
    if db is None:
        return

    try:
        # Find rooms with matching roomType
        rooms = db.collection("rooms").where("roomType", "==", room_type).get()
        
        if not rooms:
            print(f"⚠️ No room found with type: {room_type}")
            return

        # Generate all dates between checkin and checkout
        start = datetime.strptime(checkin, "%d/%m/%Y")
        end = datetime.strptime(checkout, "%d/%m/%Y")
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%d/%m/%Y"))
            current += timedelta(days=1)

        # Update each matching room
        for room in rooms:
            room.reference.update({
                "blockedDates": firestore.ArrayUnion(dates),
                "isAvailable": False if start.date() == datetime.now().date() else room.to_dict().get("isAvailable", True),
            })
            print(f"📅 Room {room.id} blocked dates updated: {dates}")

    except Exception as e:
        print(f"❌ Error updating room: {e}")


# ──────────────────────────────────────────────────────────
# 4. Full Sequence — Call After Every Confirmed Call
# ──────────────────────────────────────────────────────────
def process_completed_call(booking_data: dict):
    """
    Master function called at the end of every confirmed call.
    Runs the full Firestore write pipeline:
    1. Write booking + notification
    2. Update room blocked dates
    
    Args:
        booking_data: Complete booking dict from Gemini extraction
    """
    print("🔄 Processing completed call...")
    
    # Step 1: Write booking to Firestore
    booking_id = write_booking_to_firestore(booking_data)
    
    if booking_id:
        # Step 2: Update room availability
        update_room_in_firestore(
            room_type=booking_data.get("roomType", "Standard"),
            checkin=booking_data.get("checkIn", ""),
            checkout=booking_data.get("checkOut", ""),
        )
        
        guest_name = booking_data.get("guestName", "Guest")
        print(f"✅ Full pipeline complete for {guest_name} (booking: {booking_id})")
    else:
        print("⚠️ Pipeline incomplete — booking write failed")


# ──────────────────────────────────────────────────────────
# 5. Test Function — Trigger a Test Booking
# ──────────────────────────────────────────────────────────
def trigger_test_booking():
    """
    Creates a test booking for development/demo purposes.
    Run this from Python REPL:
        from firebase_functions import trigger_test_booking
        trigger_test_booking()
    """
    test_booking = {
        "guestName": "Amit Sharma",
        "guestPhone": "+91 98765 43210",
        "checkIn": datetime.now().strftime("%d/%m/%Y"),
        "checkOut": (datetime.now() + timedelta(days=2)).strftime("%d/%m/%Y"),
        "roomType": "Deluxe",
        "numGuests": 2,
        "pricePerNight": 3500,
        "totalPrice": 7000,
        "specialRequests": "Early check-in requested at 10 AM. Jain food preference.",
        "status": "confirmed",
        "transcript": "[Guest]: Hello, is early check-in available for tomorrow?\n[AI]: Yes, Mr. Sharma, we can arrange that. You can check in around 10 AM.\n[Guest]: Perfect. Also, we need Jain food.\n[AI]: Noted. I'll inform the kitchen. Anything else?\n[Guest]: No, thank you.",
        "callDuration": 180,
    }
    
    process_completed_call(test_booking)
    print("🧪 Test booking triggered!")


if __name__ == "__main__":
    trigger_test_booking()
