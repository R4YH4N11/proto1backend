import json
from difflib import get_close_matches
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.config import settings


class HospitalAPIClient:
    """Thin wrapper around the hospital backend REST API."""

    def __init__(self) -> None:
        self._base_url = settings.hospital_api_base_url.rstrip("/")
        self._timeout = settings.http_timeout_seconds
        self._default_headers = {"accept": "application/json"}

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self._base_url}/{path.lstrip('/')}"
        merged_headers = {**self._default_headers, **(headers or {})}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=merged_headers,
                )
                response.raise_for_status()
                if response.content:
                    return response.json()
                return {}
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Hospital API error {exc.response.status_code} for {url}: "
                f"{exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Hospital API request failed for {url}: {exc}") from exc

    def search_doctors(
        self, query: str, city: Optional[str], ai_mode: bool
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"query": query, "ai_mode": ai_mode}
        if city:
            payload["city"] = city
        return self._request(
            "POST",
            "/patient/search-doctor",
            json_body=payload,
            headers={"Content-Type": "application/json"},
        )

    def book_appointment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/patient/book-appointment",
            json_body=payload,
            headers={"Content-Type": "application/json"},
        )

    def doctor_availability_week(self, doctor_id: str) -> Dict[str, Any]:
        return self._request(
            "GET",
            f"/doctor/availability/week/{doctor_id}",
        )

    def appointments_by_phone(
        self, phone_number: str, page: int, limit: int
    ) -> Dict[str, Any]:
        safe_limit = max(1, min(limit, 50))
        return self._request(
            "GET",
            "/patient/appointments-by-phone",
            params={
                "phone_number": phone_number,
                "page": page,
                "limit": safe_limit,
            },
        )


_hospital_client = HospitalAPIClient()


SPECIALTY_SYNONYMS: Dict[str, List[str]] = {
    "Cardiologist": [
        "cardiologist",
        "heart doctor",
        "heart specialist",
        "हृदय रोग विशेषज्ञ",
        "हृदयाचा डॉक्टर",
    ],
    "Dermatologist": [
        "dermatologist",
        "skin doctor",
        "skin specialist",
        "त्वचा रोग विशेषज्ञ",
        "त्वचेचा डॉक्टर",
    ],
    "Gastroenterologist": [
        "gastroenterologist",
        "gastro",
        "stomach doctor",
        "digestive doctor",
        "पोटाचा डॉक्टर",
        "पोटाचे डॉक्टर",
        "अन्ननलिका विशेषज्ञ",
    ],
    "Pediatrician": [
        "pediatrician",
        "child doctor",
        "बालरोग तज्ञ",
        "बाळांचा डॉक्टर",
    ],
    "Neurologist": [
        "neurologist",
        "brain doctor",
        "न्यूरोलॉजिस्ट",
        "मेंदूचा डॉक्टर",
    ],
    "Orthopedic Surgeon": [
        "orthopedic",
        "bone doctor",
        "हाडांचा डॉक्टर",
        "ऑर्थोपेडिक",
    ],
}

ALIAS_TO_SPECIALTY: Dict[str, str] = {}
for canonical, aliases in SPECIALTY_SYNONYMS.items():
    for alias in aliases:
        ALIAS_TO_SPECIALTY[alias.lower()] = canonical


def _format_json(payload: Dict[str, Any]) -> str:
    if not payload:
        return "No data returned."
    return json.dumps(payload, indent=2, ensure_ascii=True)


def normalize_specialty_query(raw_query: str) -> str:
    """Map free-text specialty queries to canonical backend-friendly terms."""
    if not raw_query:
        return raw_query

    cleaned = raw_query.strip().lower()
    if not cleaned:
        return raw_query

    if cleaned in ALIAS_TO_SPECIALTY:
        return ALIAS_TO_SPECIALTY[cleaned]

    close_match = get_close_matches(
        cleaned, ALIAS_TO_SPECIALTY.keys(), n=1, cutoff=0.75
    )
    if close_match:
        return ALIAS_TO_SPECIALTY[close_match[0]]

    for alias, canonical in ALIAS_TO_SPECIALTY.items():
        if alias in cleaned:
            return canonical

    return raw_query


def _format_doctor_search_response(
    response: Dict[str, Any],
    normalized_query: str,
    original_query: str,
    city: Optional[str],
) -> str:
    status = response.get("status")
    message = response.get("message")
    doctors = response.get("doctors") or []
    suggestions = response.get("suggestions") or []
    lines: List[str] = []

    if status != "success":
        summary = message or "Doctor search failed."
        lines.append(f"Doctor search failed: {summary}")
    elif doctors:
        city_label = city or "available locations"
        lines.append(
            f"Found {len(doctors)} doctor(s) for {normalized_query} in {city_label}."
        )
        for doctor in doctors[:5]:
            name = doctor.get("full_name", "Unknown doctor")
            specialization = doctor.get("specialization", normalized_query)
            hospital = doctor.get("hospital_name", "Unknown hospital")
            location = (
                doctor.get("hospital_address") or doctor.get("location") or city_label
            )
            phone = doctor.get("phone", "Phone not listed")
            doctor_id = doctor.get("doctor_id", "ID unavailable")
            fee = doctor.get("consultation_fee")
            fee_text = (
                f"Fee: INR {fee:.0f}"
                if isinstance(fee, (int, float))
                else "Fee not listed"
            )
            lines.append(
                f"- {name} ({specialization}) at {hospital}, {location}. "
                f"Phone: {phone}. Doctor ID: {doctor_id}. {fee_text}."
            )
        if len(doctors) > 5:
            lines.append(
                f"There are {len(doctors) - 5} more doctor(s) available. "
                "Ask if you would like the full list."
            )
        if suggestions:
            lines.append("General care tips:")
            for tip in suggestions[:3]:
                lines.append(f"* {tip}")
    else:
        city_label = city or "the requested area"
        fallback = message or "No matching doctors returned by the search API."
        lines.append(f"No doctors found for {normalized_query} in {city_label}.")
        lines.append(f"API message: {fallback}")

    if normalized_query.lower() != original_query.lower():
        lines.append(
            f"(Original query '{original_query}' was interpreted as '{normalized_query}'.)"
        )

    return "\n".join(lines)


def _format_appointments_response(
    response: Any, phone_number: str, page: int, limit: int
) -> str:
    lines: List[str] = []
    appointments: Optional[List[Dict[str, Any]]] = None
    api_message: Optional[str] = None

    if isinstance(response, list):
        appointments = response
    elif isinstance(response, dict):
        for key in ("appointments", "data", "results", "items"):
            value = response.get(key)
            if isinstance(value, list):
                appointments = value
                break
            if isinstance(value, dict) and isinstance(value.get("appointments"), list):
                appointments = value.get("appointments")
                break
        api_message = response.get("message")

    if not appointments:
        fallback = api_message or "No appointments returned by the API."
        lines.append(
            f"No appointments found for phone number {phone_number} (page {page}, limit {limit})."
        )
        lines.append(f"API message: {fallback}")
        return "\n".join(lines)

    lines.append(
        f"Found {len(appointments)} appointment(s) linked to {phone_number} (page {page}, limit {limit})."
    )
    for appointment in appointments[:5]:
        appointment_id = (
            appointment.get("appointment_id")
            or appointment.get("id")
            or "Unknown appointment ID"
        )
        doctor = appointment.get("doctor") or {}
        doctor_name = (
            appointment.get("doctor_name")
            or doctor.get("full_name")
            or doctor.get("name")
            or "Unknown doctor"
        )
        doctor_id = (
            appointment.get("doctor_id")
            or doctor.get("doctor_id")
            or doctor.get("id")
            or "Unknown doctor ID"
        )
        meeting_time = (
            appointment.get("meeting_time")
            or appointment.get("appointment_time")
            or appointment.get("scheduled_time")
            or "Unknown time"
        )
        status = appointment.get("status", "status unavailable")
        appointment_type = appointment.get("appointment_type") or appointment.get(
            "type", "type unavailable"
        )
        location = (
            appointment.get("location")
            or appointment.get("city")
            or doctor.get("hospital_name")
            or doctor.get("location")
            or "Location not provided"
        )
        line_parts: List[str] = [f"- Appointment {appointment_id}"]
        if doctor_name != "Unknown doctor":
            doctor_fragment = f" with {doctor_name}"
            if doctor_id != "Unknown doctor ID":
                doctor_fragment += f" (Doctor ID: {doctor_id})"
            line_parts.append(doctor_fragment)
        line_parts.append(f" on {meeting_time}.")
        line_parts.append(f" Status: {status}.")
        if appointment_type and appointment_type != "type unavailable":
            line_parts.append(f" Type: {appointment_type}.")
        if location and location != "Location not provided":
            line_parts.append(f" Location: {location}.")
        lines.append("".join(line_parts))

    if len(appointments) > 5:
        lines.append(
            f"There are {len(appointments) - 5} additional appointment(s). "
            "Request a higher limit or another page to see more."
        )

    if api_message:
        lines.append(f"API message: {api_message}")

    return "\n".join(lines)


class SearchDoctorInput(BaseModel):
    query: str = Field(..., description="Doctor name or specialty to search for.")
    city: Optional[str] = Field(
        default=None,
        description="City to filter the doctor search results.",
    )
    ai_mode: bool = Field(
        default=True,
        description="Whether to enable AI mode on the backend search endpoint.",
    )


def search_doctors_tool(
    query: str, city: Optional[str] = None, ai_mode: bool = True
) -> str:
    """Return a formatted list of doctors that match the search criteria."""
    normalized_query = normalize_specialty_query(query)
    response = _hospital_client.search_doctors(
        query=normalized_query, city=city, ai_mode=ai_mode
    )
    return _format_doctor_search_response(
        response=response,
        normalized_query=normalized_query,
        original_query=query,
        city=city,
    )


class DoctorAvailabilityInput(BaseModel):
    doctor_id: str = Field(..., description="Doctor UUID to fetch weekly availability.")


def doctor_availability_tool(doctor_id: str) -> str:
    """Fetch a doctor's weekly availability calendar."""
    response = _hospital_client.doctor_availability_week(doctor_id=doctor_id)
    return _format_json(response)


class AppointmentsByPhoneInput(BaseModel):
    phone_number: str = Field(..., description="Patient's phone number.")
    page: int = Field(default=1, ge=1, description="Results page to fetch.")
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum appointments to return per page (max 50).",
    )


def appointments_by_phone_tool(
    phone_number: str, page: int = 1, limit: int = 10
) -> str:
    """Retrieve appointments associated with a patient's phone number."""
    response = _hospital_client.appointments_by_phone(
        phone_number=phone_number, page=page, limit=limit
    )
    return _format_appointments_response(response, phone_number, page, limit)


class BookAppointmentInput(BaseModel):
    doctor_id: str = Field(..., description="Doctor UUID to book an appointment with.")
    patient_name: str = Field(..., description="Patient full name.")
    patient_phone_number: str = Field(..., description="Patient's contact number.")
    meeting_time: str = Field(
        ...,
        description="ISO timestamp for the appointment, e.g. 2025-10-09T11:20:03.544Z.",
    )
    appointment_type: str = Field(
        default="online",
        description="Type of appointment, e.g. online or in_person.",
    )
    status: str = Field(
        default="scheduled",
        description="Initial appointment status, e.g. scheduled.",
    )
    client_id: Optional[str] = Field(
        default=None,
        description="Client identifier. Falls back to HOSPITAL_CLIENT_ID when omitted.",
    )


def book_appointment_tool(
    doctor_id: str,
    patient_name: str,
    patient_phone_number: str,
    meeting_time: str,
    appointment_type: str = "online",
    status: str = "scheduled",
    client_id: Optional[str] = None,
) -> str:
    """Create a new appointment for a patient with the specified doctor."""
    resolved_client_id = client_id or settings.require_hospital_client_id()
    payload = {
        "doctor_id": doctor_id,
        "patient_name": patient_name,
        "patient_phone_number": patient_phone_number,
        "client_id": resolved_client_id,
        "meeting_time": meeting_time,
        "appointment_type": appointment_type,
        "status": status,
    }
    response = _hospital_client.book_appointment(payload=payload)
    return _format_json(response)


def get_hospital_tools() -> List[StructuredTool]:
    """Expose the live hospital API endpoints as LangChain tools."""
    return [
        StructuredTool.from_function(
            func=search_doctors_tool,
            name="search_doctors",
            description=(
                "Search for doctors or specialists by name or specialty, optionally "
                "filtering by city."
            ),
            args_schema=SearchDoctorInput,
        ),
        StructuredTool.from_function(
            func=doctor_availability_tool,
            name="doctor_weekly_availability",
            description=(
                "Retrieve a doctor's weekly availability slots using their UUID."
            ),
            args_schema=DoctorAvailabilityInput,
        ),
        StructuredTool.from_function(
            func=appointments_by_phone_tool,
            name="appointments_by_phone",
            description=(
                "List appointments for a patient identified by phone number. "
                "Supports pagination."
            ),
            args_schema=AppointmentsByPhoneInput,
        ),
        StructuredTool.from_function(
            func=book_appointment_tool,
            name="book_appointment",
            description=(
                "Schedule a new appointment for a patient with a doctor. "
                "Requires doctor ID, patient details, meeting time, and uses the "
                "configured hospital client identifier when not provided."
            ),
            args_schema=BookAppointmentInput,
        ),
    ]
