

# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — TRACK A FORENSICS (Semantic Knowledge Graph & Evidence-Based Reasoning)
# ══════════════════════════════════════════════════════════════════════════════
with tab9:
    st.markdown("""
    <div class="acmgs-header">
        <h1>🔬 Track A Forensics</h1>
        <p>Semantic reasoning & evidence-based batch analysis</p>
        <span class="hbadge hbadge-green">KNOWLEDGE GRAPH</span>
        <span class="hbadge hbadge-blue">BATCH FORENSICS</span>
        <span class="hbadge hbadge-green">EVIDENCE-CITATIONS</span>
    </div>
    """, unsafe_allow_html=True)
    
    # ─── Initialize semantic graph ──────────────────────────────────────
    try:
        @st.cache_resource
        def get_semantic_graph():
            return SemanticGraph()
        
        sg = get_semantic_graph()
        batch_forensics = BatchForensics(sg)
        evidence_rec = EvidenceRecommender(sg)
    except Exception as e:
        st.error(f"Failed to load semantic layer: {e}")
        st.stop()
    
    # ─── SECTION 1: BATCH SELECTION & FORENSICS ────────────────────────
    st.markdown('<div class="slabel">🔍 Batch Forensics Analysis</div>', unsafe_allow_html=True)
    
    col_for1, col_for2 = st.columns([3, 1])
    
    with col_for1:
        batch_options = [str(b) for b in df_batches["batch_id"].head(100).tolist()]
        selected_batch = st.selectbox(
            "Select a batch for detailed forensics analysis:",
            batch_options,
            help="Choose a batch to analyze its performance and relationships"
        )
    
    with col_for2:
        analysis_type = st.radio(
            "Analysis Type:",
            ["Forensics", "Evidence", "Comparison"],
            horizontal=True
        )
    
    if selected_batch:
        # Perform analysis based on type
        if analysis_type == "Forensics":
            forensics = batch_forensics.analyze_batch(selected_batch)
            
            st.markdown("#### Why-Why Analysis")
            st.markdown(forensics.get("narrative", "Analysis unavailable"))
            
            # Display why questions in expandable sections
            if "why_questions" in forensics:
                for i, why_q in enumerate(forensics["why_questions"], 1):
                    with st.expander(f"Q{i}: {why_q.get('question', '')}"):
                        st.write(f"**Answer:** {why_q.get('answer', 'N/A')}")
                        if why_q.get('relationship'):
                            st.json(why_q['relationship'])
        
        elif analysis_type == "Evidence":
            recs = evidence_rec.recommend_batch_optimization(selected_batch)
            report = evidence_rec.format_recommendation_report(selected_batch)
            
            st.markdown("#### Evidence-Based Recommendations")
            st.markdown(report, unsafe_allow_html=True)
            
            # Display as cards
            if recs:
                for rec in recs:
                    with st.expander(
                        f"{rec['title']} — {rec['confidence']} Confidence",
                        expanded=True
                    ):
                        col_e1, col_e2 = st.columns([2, 1])
                        
                        with col_e1:
                            st.markdown(f"**Impact:** {rec['impact']}")
                            st.markdown("**Evidence Sources:**")
                            for ev in rec["evidence"]:
                                st.markdown(f"  • `{ev['type']}` — {ev['detail']}")
                            
                            st.markdown("**Reasoning Chain:**")
                            for step in rec["reasoning"]:
                                st.markdown(f"  {step}")
                        
                        with col_e2:
                            # Confidence badge
                            colors = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
                            st.markdown(f"{colors.get(rec['confidence'], '⚪')} **{rec['confidence']}**")
        
        elif analysis_type == "Comparison":
            root_causes = batch_forensics.trace_root_cause(selected_batch)
            st.markdown("#### Root Cause Analysis")
            st.markdown(root_causes.get("narrative", "Analysis unavailable"))
            
            if root_causes.get("root_causes"):
                st.markdown("**Identified Root Causes:**")
                for cause in root_causes["root_causes"]:
                    st.markdown(f"""
                    **{cause['cause']}**
                    - Impact: {cause['impact']}
                    - Evidence: {cause['evidence']}
                    - Confidence: {cause['confidence']*100:.0f}%
                    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ─── SECTION 2: SEMANTIC RELATIONSHIPS ──────────────────────────────
    st.markdown('<div class="slabel">🔗 Semantic Relationships (Track A)</div>', unsafe_allow_html=True)
    
    rel_view = st.selectbox(
        "View relationships:",
        ["All", "Asset-Produces-Batch", "Batch-Has-Energy", "Material-Affects-Yield", "Anomaly-Triggered-By-Drift"],
        help="Filter relationships by type"
    )
    
    if rel_view == "All":
        all_rels = sg.get_relationships()
        df_rels = pd.DataFrame([{
            "Source": r["source"][:20],
            "Relation": r["relation_type"],
            "Target": r["target"][:20],
            "Confidence": f"{r['confidence']*100:.0f}%",
        } for r in all_rels[:50]])
        
        st.dataframe(df_rels, use_container_width=True, height=400)
    
    else:
        rel_types = {
            "Asset-Produces-Batch": "Produces",
            "Batch-Has-Energy": "Has",
            "Material-Affects-Yield": "affects",
            "Anomaly-Triggered-By-Drift": "triggered_by",
        }
        rels = sg.get_relationships(relation_type=rel_types[rel_view])
        df_rels = pd.DataFrame([{
            "Source": r["source"][:25],
            "Target": r["target"][:25],
            "Evidence": r["evidence"][:40],
            "Confidence": f"{r['confidence']*100:.0f}%",
        } for r in rels[:30]])
        
        st.dataframe(df_rels, use_container_width=True, height=400)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ─── SECTION 3: KNOWLEDGE GRAPH STATISTICS ──────────────────────────
    st.markdown('<div class="slabel">📊 Knowledge Graph Statistics</div>', unsafe_allow_html=True)
    
    # Count relationships by type
    all_rels = sg.get_relationships()
    rel_counts = {}
    for rel in all_rels:
        rel_type = rel["relation_type"]
        rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
    
    col_kg1, col_kg2, col_kg3, col_kg4 = st.columns(4)
    
    with col_kg1:
        st.metric("Total Relationships", len(all_rels))
    
    with col_kg2:
        st.metric("Unique Entities", len(sg.entities))
    
    with col_kg3:
        avg_confidence = np.mean([r["confidence"] for r in all_rels]) if all_rels else 0
        st.metric("Avg Confidence", f"{avg_confidence*100:.0f}%")
    
    with col_kg4:
        st.metric("Relationship Types", len(rel_counts))
    
    # Relationship type distribution
    st.markdown("**Relationship Type Distribution:**")
    fig_rel = go.Figure(data=[
        go.Bar(x=list(rel_counts.keys()), y=list(rel_counts.values()), marker_color="#00d4ff")
    ])
    fig_rel.update_layout(
        title="Relationships by Type",
        xaxis_title="Relationship Type",
        yaxis_title="Count",
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=50, b=80),
        showlegend=False,
    )
    st.plotly_chart(fig_rel, use_container_width=True, config={"displayModeBar": False})
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ─── SECTION 4: TRACK A RELATIONSHIPS SUMMARY ───────────────────────
    st.markdown('<div class="slabel">📋 Track A Requirements Status</div>', unsafe_allow_html=True)
    
    track_a_components = [
        {
            "relationship": "Asset –[Produces]→ Batch",
            "status": "✅ Implemented",
            "details": "Manufacturing assets linked to produced batches",
        },
        {
            "relationship": "Batch –[Has]→ Energy Patterns",
            "status": "✅ Implemented",
            "details": "LSTM energy embeddings extracted per batch",
        },
        {
            "relationship": "Process Parameters –[influences]→ Energy Patterns",
            "status": "✅ Implemented",
            "details": "Parameter relationships model energy impact",
        },
        {
            "relationship": "Batch –[compared_against]→ Golden Signature",
            "status": "✅ Implemented",
            "details": "Pareto-optimal solutions serve as golden references",
        },
        {
            "relationship": "Golden Signature –[optimized_for]→ Objectives",
            "status": "✅ Implemented",
            "details": "4-objective optimization: Yield, Quality, Energy, Carbon",
        },
        {
            "relationship": "Raw Material –[affects]→ Yield Outcome",
            "status": "✅ Implemented",
            "details": "Material grade impacts yield prediction",
        },
        {
            "relationship": "Energy Patterns –[indicates]→ Asset Health",
            "status": "✅ Implemented",
            "details": "Reconstruction error indicates anomalies",
        },
        {
            "relationship": "Anomaly –[triggered_by]→ Process Drift",
            "status": "✅ Implemented",
            "details": "Process deviations trigger energy anomalies",
        },
    ]
    
    df_track_a = pd.DataFrame(track_a_components)
    st.dataframe(df_track_a, use_container_width=True)
    
    st.success("🎉 **All Track A relationships successfully implemented as a semantic layer**")
    st.info("ℹ️ Semantic relationships are mapped from existing ACMGS database and ML models without requiring external graph databases.")
