import json
import requests
import streamlit as st


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

API_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(
    page_title="Policy-Aware Claim Decision Engine",
    page_icon="🏥",
    layout="wide"
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🏥 Policy-Aware Claim Decision Engine")

st.write(
    "Analyze a synthetic health-insurance claim using "
    "policy-grounded multi-agent RAG."
)


# ---------------------------------------------------------
# Example claim
# ---------------------------------------------------------

example_claim = {
    "case_id": "PUB-001",
    "policy_start_date": "2025-01-01",
    "claim_date": "2026-03-14",
    "coverage_duration_months": 14,
    "claim_type": "Hospitalization",
    "treatment": "Acute appendicitis requiring appendectomy",
    "hospitalization_type": "inpatient",
    "hospitalization_hours": 96,
    "network_provider": True,
    "documentation_available": [
        "claim_form",
        "discharge_summary",
        "hospital_bill",
        "diagnostic_reports"
    ]
}


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("Claim Input")

input_mode = st.sidebar.radio(
    "Choose input method",
    ["Example Case", "Paste JSON"]
)


if input_mode == "Example Case":

    claim = example_claim

    st.sidebar.success(
        "Using example case: PUB-001"
    )

else:

    json_text = st.sidebar.text_area(
        "Paste claim JSON",
        value=json.dumps(
            example_claim,
            indent=2
        ),
        height=400
    )

    try:

        claim = json.loads(json_text)

    except json.JSONDecodeError:

        st.error(
            "Invalid JSON. Please enter valid claim JSON."
        )

        st.stop()


# ---------------------------------------------------------
# Analyze button
# ---------------------------------------------------------

if st.button(
    "Analyze Claim",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Running multi-agent policy analysis..."
    ):

        try:

            response = requests.post(
                API_URL,
                json={
                    "claim": claim
                },
                timeout=300
            )

        except requests.exceptions.RequestException as error:

            st.error(
                "Could not connect to the backend API."
            )

            st.info(
                "Make sure FastAPI is running at "
                "http://127.0.0.1:8000"
            )

            st.code(str(error))

            st.stop()


    # -----------------------------------------------------
    # API error handling
    # -----------------------------------------------------

    if response.status_code != 200:

        st.error(
            f"API Error {response.status_code}"
        )

        st.code(response.text)

        st.stop()


    result = response.json()


    # -----------------------------------------------------
    # Decision
    # -----------------------------------------------------

    decision = result.get(
        "decision",
        "UNKNOWN"
    )

    confidence = result.get(
        "confidence",
        0
    )


    st.divider()

    st.subheader("Decision")


    if decision == "ADMISSIBLE":

        st.success(
            f"✅ {decision}"
        )

    elif decision == "ADMISSIBLE_WITH_LIMITS":

        st.warning(
            f"⚠️ {decision}"
        )

    elif decision == "PARTIALLY_ADMISSIBLE":

        st.warning(
            f"⚠️ {decision}"
        )

    elif decision == "NOT_ADMISSIBLE":

        st.error(
            f"❌ {decision}"
        )

    elif decision == "NEEDS_REVIEW":

        st.warning(
            "🔎 NEEDS_REVIEW"
        )

        st.info(
            "The available evidence is insufficient "
            "for a final decision."
        )

    else:

        st.warning(
            f"Decision: {decision}"
        )


    st.metric(
        "Confidence",
        f"{confidence:.0%}"
    )


    # -----------------------------------------------------
    # Key findings
    # -----------------------------------------------------

    st.divider()

    st.subheader("Key Findings")

    findings = result.get(
        "key_findings",
        []
    )


    if findings:

        for finding in findings:

            if isinstance(
                finding,
                dict
            ):

                finding_type = finding.get(
                    "type",
                    "FINDING"
                )

                statement = finding.get(
                    "statement",
                    ""
                )

                st.markdown(
                    f"**{finding_type}:** {statement}"
                )

            else:

                st.write(
                    f"• {finding}"
                )

    else:

        st.write(
            "No key findings returned."
        )


    # -----------------------------------------------------
    # Applicable limits
    # -----------------------------------------------------

    limits = result.get(
        "applicable_limits",
        []
    )


    if limits:

        st.divider()

        st.subheader(
            "Applicable Limits"
        )

        for limit in limits:

            st.write(
                f"• {limit}"
            )


    # -----------------------------------------------------
    # Missing evidence
    # -----------------------------------------------------

    missing_evidence = result.get(
        "missing_evidence",
        []
    )


    if missing_evidence:

        st.divider()

        st.subheader(
            "Missing Evidence"
        )

        for evidence in missing_evidence:

            st.write(
                f"• {evidence}"
            )


    # -----------------------------------------------------
    # Policy citations
    # -----------------------------------------------------

    citations = result.get(
        "citations",
        []
    )


    if citations:

        st.divider()

        st.subheader(
            "Policy Evidence & Citations"
        )


        for index, citation in enumerate(
            citations,
            start=1
        ):

            page = citation.get(
                "page",
                "N/A"
            )

            section = citation.get(
                "section",
                "N/A"
            )

            source = citation.get(
                "source",
                "policy"
            )

            chunk_id = citation.get(
                "chunk_id",
                "N/A"
            )


            with st.expander(
                f"Evidence {index} — Page {page}"
            ):

                st.write(
                    f"**Source:** {source}"
                )

                st.write(
                    f"**Page:** {page}"
                )

                st.write(
                    f"**Section:** {section}"
                )

                st.write(
                    f"**Chunk ID:** {chunk_id}"
                )


    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    validation = result.get(
        "validation",
        {}
    )

    validation_status = validation.get(
        "validation_status",
        "UNKNOWN"
    )


    st.divider()

    st.subheader(
        "Validation"
    )


    if validation_status == "PASS":

        st.success(
            "✅ Validation: PASS"
        )

    else:

        st.error(
            f"❌ Validation: {validation_status}"
        )


    # -----------------------------------------------------
    # Validation details
    # -----------------------------------------------------

    unsupported_claims = validation.get(
        "unsupported_claims",
        []
    )


    if unsupported_claims:

        st.warning(
            "Unsupported Claims"
        )

        for claim_item in unsupported_claims:

            st.write(
                f"• {claim_item}"
            )


    # -----------------------------------------------------
    # Agent trace
    # -----------------------------------------------------

    trace = result.get(
        "trace",
        []
    )


    if trace:

        st.divider()

        st.subheader(
            "Agent Trace"
        )


        for step in trace:

            agent = step.get(
                "agent",
                "Agent"
            )

            action = step.get(
                "action",
                ""
            )


            with st.expander(
                agent
            ):

                st.write(
                    action
                )


                if "query_count" in step:

                    st.write(
                        f"Retrieval queries: "
                        f"{step['query_count']}"
                    )


                if "evidence_count" in step:

                    st.write(
                        f"Evidence chunks: "
                        f"{step['evidence_count']}"
                    )


                if "validation_status" in step:

                    st.write(
                        f"Validation status: "
                        f"{step['validation_status']}"
                    )