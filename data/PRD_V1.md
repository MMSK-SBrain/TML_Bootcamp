# Product Requirements Document (PRD)

## AI Vehicle Diagnostic Assistant – MVP (Web, Voice-First)

---

# 1. Purpose

Build a **simple web-based, voice-first AI assistant** for **non-technical drivers** to describe vehicle problems and receive **safe, non-technical, structured guidance** using an LLM with RAG grounding.

This MVP validates:

• Voice-driven interaction
• Knowledge-grounded responses
• Safety-bounded outputs
• Clear, human-friendly diagnostic guidance

This is an **internal product MVP**.

---

# 2. Target User

Primary user:
**Non-technical vehicle driver**

Assumptions:
• Does not understand automotive systems
• Describes problems emotionally and vaguely
• Wants clarity, seriousness level, and next steps
• Must not be given mechanical instructions

---

# 3. Platform & Access

• Platform: Web application
• Devices: Mobile browser + desktop browser
• No login
• No persistent user profiles
• English only (MVP)
• Microphone permission required

---

# 4. High-Level User Journey

1. User opens web page
2. Clicks microphone button
3. Describes vehicle problem
4. Sees live transcription
5. Submits query
6. AI processes input using RAG
7. User receives structured guidance
8. Optional: hears spoken response
9. User may ask a follow-up question

---

# 5. Core User Flows

---

## Flow 1: Voice-Based Problem Description (Primary)

1. User taps “Start speaking”
2. App shows:
   • Active mic indicator
   • Live transcript
3. User taps “Stop”
4. Transcript becomes editable
5. User clicks “Analyze problem”
6. Loading state shown
7. Structured response displayed

---

## Flow 2: Text Fallback

1. User types into input box
2. Clicks “Analyze problem”
3. Same processing and output flow

---

## Flow 3: Follow-up Clarification

1. User asks additional question
2. Previous context retained
3. Updated structured response generated

---

## Flow 4: Low-Knowledge / Unsupported Case

1. User submits issue
2. System cannot confidently retrieve knowledge
3. UI displays:
   • Low coverage message
   • Safety guidance
   • Escalation recommendation

---

# 6. Web UI Requirements

---

## 6.1 Layout (Single Page)

### Header

• Product name
• Short instruction text
“Describe what is happening with your vehicle.”

---

### Input Zone

• Large microphone button
• Status indicator:

* Listening
* Processing
* Ready

• Live speech transcript
• Editable text box
• Buttons:

* Start speaking
* Stop
* Analyze problem
* Reset

---

### Output Zone (Structured Cards)

Each response must render **six fixed blocks**:

1. What this likely means
2. How serious this could be
3. Common possible reasons
4. What you can safely do now
5. When to escalate
6. Knowledge coverage indicator

Each block should be a **separate visual card**.

---

### Audio Output

• “Play response” button
• Adjustable volume
• Visible speaking animation

---

### Footer

• Safety disclaimer
• Microphone permission status

---

## 6.2 Visual Design Principles

• Extremely simple
• High contrast
• Large buttons
• Mobile-friendly
• Minimal scrolling
• Calm, reassuring tone

---

# 7. Interaction Design

---

## Microphone Interaction

States:

Idle → Listening → Processing → Responding → Idle

User must always see:
• Current state
• Ability to stop recording
• Transcript preview

---

## Loading State

• Spinner
• Text: “Analyzing your vehicle issue…”
• Disable inputs

---

## Error States

• Microphone denied
• No speech detected
• AI failure
• Low confidence response

Each error must include:
• Simple explanation
• Clear next step

---

# 8. Response Contract (Strict)

The LLM must always return:

```
{
  "meaning": "...",
  "severity": "...",
  "possible_causes": ["...", "..."],
  "safe_actions": ["...", "..."],
  "escalation": "...",
  "knowledge_coverage": "high | medium | low"
}
```

Frontend must render each field in its own section.

---

# 9. Functional Requirements

---

## Must Have

• Voice capture
• Speech-to-text
• Editable transcript
• Single-call AI backend
• RAG integration
• Structured response rendering
• Safety language
• Follow-up questioning
• Mobile support

---

## Should Have

• Audio playback
• Conversation continuity
• Clear severity indicators
• Confidence/coverage display

---

## Explicitly Out of Scope (MVP)

• Vehicle OBD data
• VIN decoding
• Repair instructions
• Cost estimation
• Accounts
• History
• Booking
• Workshop tools

---

# 10. Safety & Compliance Rules

The system must always:

• Avoid mechanical instructions
• Avoid certainty language
• Escalate all high-risk topics
• Use non-technical vocabulary
• Insert safety messaging for:

* Brakes
* Steering
* Smoke/fire
* EV battery
* Overheating
* Airbags

---

# 11. Technical MVP Mapping

Frontend:
• HTML/CSS/React/Vue
• Web Speech API (optional)
• REST calls

Backend:
• STT API
• LLM API
• RAG retriever
• Vector DB

Processing:
User → STT → LLM + RAG → Safety prompt → JSON → UI

---

# 12. Acceptance Criteria

The MVP is considered successful if:

• A non-technical user can operate it without instruction
• Responses are structured and understandable
• Dangerous issues always trigger escalation
• The system gracefully handles unknown problems
• It can be demoed end-to-end live

---

# 13. Suggested Next Artifacts (optional)

If you want, I can quickly produce:

• Wireframe-level UI layout
• API contract spec
• Prompt system design
• Sample conversations
• Frontend task breakdown

If you want this converted into a **Notion-style PRD**, **engineering ticket format**, or **Figma-ready UI spec**, tell me which format you prefer.
