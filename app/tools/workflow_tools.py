"""
Workflow routing tools for complex hybrid workflows
"""

def route_workflow(decision_key: str, comprehensive_route: list, alternative_route: list, **context) -> dict:
    """
    Route workflow execution based on decision criteria
    
    Args:
        decision_key: The context key containing the routing decision
        comprehensive_route: List of nodes for comprehensive analysis
        alternative_route: List of nodes for alternative analysis
        **context: Workflow context containing decision data
    
    Returns:
        dict: Routing decision and metadata
    """
    
    # Get the decision from context
    decision = context.get(decision_key, "")
    analysis_route = context.get("analysis_route", "")
    data_quality_score = context.get("data_quality_score", 5)
    
    # Default routing logic
    route_decision = "alternative"
    reasoning = "Default routing to alternative path"
    
    # Determine routing based on decision content
    if "comprehensive" in decision.lower() or data_quality_score >= 7:
        route_decision = "comprehensive"
        reasoning = f"High quality data (score: {data_quality_score}) - routing to comprehensive analysis"
    elif "alternative" in decision.lower() or data_quality_score < 7:
        route_decision = "alternative" 
        reasoning = f"Moderate quality data (score: {data_quality_score}) - routing to alternative analysis"
    
    # Handle explicit analysis route
    if analysis_route:
        route_decision = analysis_route
        reasoning = f"Explicit routing decision: {analysis_route}"
    
    return {
        "route_decision": route_decision,
        "routing_reasoning": reasoning,
        "comprehensive_nodes": comprehensive_route if route_decision == "comprehensive" else [],
        "alternative_nodes": alternative_route if route_decision == "alternative" else [],
        "data_quality_score": data_quality_score,
        "decision_source": decision_key
    }


def validate_workflow_phase(phase_id: str, required_context: list, **context) -> dict:
    """
    Validate that a workflow phase has the required context to proceed
    
    Args:
        phase_id: Identifier for the workflow phase
        required_context: List of context keys required for this phase
        **context: Current workflow context
    
    Returns:
        dict: Validation results
    """
    
    missing_context = []
    available_context = []
    
    for key in required_context:
        if key in context and context[key] is not None:
            available_context.append(key)
        else:
            missing_context.append(key)
    
    is_valid = len(missing_context) == 0
    confidence = len(available_context) / len(required_context) if required_context else 1.0
    
    return {
        "phase_id": phase_id,
        "is_valid": is_valid,
        "confidence": confidence,
        "missing_context": missing_context,
        "available_context": available_context,
        "validation_status": "passed" if is_valid else "failed",
        "can_proceed": confidence >= 0.7  # Proceed if at least 70% context available
    }


def aggregate_analysis_results(financial_analysis: dict = None, risk_assessment: dict = None, 
                             strategy_plan: dict = None, alternative_analysis: dict = None,
                             **context) -> dict:
    """
    Aggregate results from different analysis paths
    
    Args:
        financial_analysis: Results from financial analysis
        risk_assessment: Results from risk assessment  
        strategy_plan: Results from strategy planning
        alternative_analysis: Results from alternative analysis
        **context: Additional context
    
    Returns:
        dict: Aggregated analysis results
    """
    
    analysis_type = "unknown"
    confidence_score = 0.0
    key_findings = []
    recommendations = []
    
    # Determine analysis type and aggregate results
    if financial_analysis or risk_assessment or strategy_plan:
        analysis_type = "comprehensive"
        components = []
        
        if financial_analysis:
            components.append("financial")
            key_findings.extend(financial_analysis.get("key_points", []))
            
        if risk_assessment:
            components.append("risk")
            key_findings.extend(risk_assessment.get("key_points", []))
            
        if strategy_plan:
            components.append("strategic")
            key_findings.extend(strategy_plan.get("key_points", []))
            
        confidence_score = 0.9  # High confidence for comprehensive analysis
        
    elif alternative_analysis:
        analysis_type = "alternative"
        key_findings = alternative_analysis.get("key_points", [])
        confidence_score = alternative_analysis.get("confidence_level", 0.6)
        
    return {
        "analysis_type": analysis_type,
        "confidence_score": confidence_score,
        "key_findings": key_findings[:10],  # Limit to top 10 findings
        "recommendations": recommendations,
        "aggregation_timestamp": context.get("timestamp", ""),
        "source_components": components if analysis_type == "comprehensive" else ["alternative"],
        "is_complete": confidence_score >= 0.5
    }
