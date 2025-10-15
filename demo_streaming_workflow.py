#!/usr/bin/env python3
"""
Comprehensive streaming workflow demo
"""

import requests
import json
import time
from datetime import datetime

def demo_streaming_workflow():
    """Demonstrate the full streaming workflow capabilities"""
    
    # Test with a more specific task to get better responses
    payload = {
        'workflow_id': 'sequential_research_workflow',
        'input_data': {
            'task': 'Research the benefits of solar energy for residential use',
            'prompt': 'Research residential solar energy benefits including cost savings, environmental impact, and energy independence. Provide specific data and analysis.'
        },
        'context': {
            'topic': 'Residential Solar Energy',
            'focus_areas': ['cost savings', 'environmental benefits', 'energy independence'],
            'target_audience': 'homeowners',
            'research_depth': 'comprehensive'
        }
    }
    
    print("🌟 COMPREHENSIVE STREAMING WORKFLOW DEMO")
    print("=" * 80)
    print(f"📋 Workflow: {payload['workflow_id']}")
    print(f"🎯 Task: {payload['input_data']['task']}")
    print(f"🔍 Focus: {', '.join(payload['context']['focus_areas'])}")
    print("=" * 80)
    
    try:
        response = requests.post(
            'http://localhost:8000/workflows/execute-stream',
            json=payload,
            stream=True,
            headers={'Accept': 'text/event-stream'},
            timeout=120
        )
        
        if response.status_code != 200:
            print(f"❌ Error: HTTP {response.status_code}")
            print(response.text)
            return
        
        print("✅ Streaming connection established")
        print("📡 REAL-TIME WORKFLOW EXECUTION:")
        print("-" * 80)
        
        # Parse and display streaming events
        current_event = None
        start_time = time.time()
        node_times = {}
        
        for line in response.iter_lines(decode_unicode=True):
            line = line.strip()
            if line:
                if line.startswith('event: '):
                    current_event = line[7:]
                elif line.startswith('data: '):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        # Add the event type to the data for processing
                        if current_event:
                            data['event_type'] = current_event
                        display_event(current_event, data, start_time, node_times)
                        current_event = None  # Reset after processing
                    except json.JSONDecodeError:
                        print(f"📄 Raw: {data_str}")
                elif line.startswith(': ping'):
                    # Ignore ping events
                    continue
        
        print("\n" + "=" * 80)
        print("🎉 STREAMING WORKFLOW DEMO COMPLETED!")
        print("\n📊 SUMMARY:")
        print(f"   ⏱️  Total execution time: {time.time() - start_time:.1f}s")
        print(f"   🔗 Node execution times:")
        for node, exec_time in node_times.items():
            print(f"      └── {node}: {exec_time:.2f}s")
        print("\n✨ Key Benefits of Streaming Workflows:")
        print("   🚀 Real-time progress monitoring")
        print("   📊 Live execution metrics")
        print("   🔍 Immediate visibility into agent activities")
        print("   ⚡ Better user experience for long-running workflows")
        print("   🛠️  Early error detection and debugging")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def display_event(event_type, data, start_time, node_times):
    """Display streaming events in a formatted way"""
    
    current_time = time.time()
    elapsed = current_time - start_time
    timestamp = data.get('timestamp', '')
    
    # Format time
    time_str = f"[{elapsed:6.1f}s]"
    
    if event_type == 'workflow_started':
        print(f"🏁 {time_str} Workflow started: {data.get('workflow_id')}")
        
    elif event_type == 'workflow_metadata':
        print(f"📋 {time_str} Workflow: {data.get('name')}")
        print(f"   {'':10} └── Topology: {data.get('topology')}")
        print(f"   {'':10} └── Nodes: {data.get('total_nodes')} ({', '.join(data.get('node_names', []))})")
        
    elif event_type == 'node_started':
        progress = data.get('progress', '')
        print(f"\n🔧 {time_str} Starting: {data.get('node_name')} ({progress})")
        print(f"   {'':10} └── Node ID: {data.get('node_id')}")
        print(f"   {'':10} └── Type: {data.get('node_type')}")
        
    elif event_type == 'agent_preparing':
        print(f"🤖 {time_str} Preparing agent...")
        print(f"   {'':10} └── Config: {data.get('agent_config')}")
        
    elif event_type == 'agent_prompt':
        prompt = data.get('prompt', '')[:80] + "..." if len(data.get('prompt', '')) > 80 else data.get('prompt', '')
        print(f"💬 {time_str} Agent prompt ready")
        print(f"   {'':10} └── {prompt}")
        
    elif event_type == 'agent_executing':
        print(f"⚡ {time_str} Agent executing... ")
        
    elif event_type == 'node_completed':
        node_id = data.get('node_id')
        exec_time = data.get('execution_time', 0)
        node_times[node_id] = exec_time
        
        progress = data.get('progress', '')
        print(f"✅ {time_str} Completed: {data.get('node_name')} ({progress})")
        print(f"   {'':10} └── Execution time: {exec_time:.2f}s")
        print(f"   {'':10} └── Status: {data.get('status')}")
        
        preview = data.get('result_preview', '')
        if preview:
            # Show a nice preview
            if len(preview) > 120:
                preview = preview[:120] + "..."
            print(f"   {'':10} └── Preview: {preview}")
            
    elif event_type == 'node_failed':
        print(f"❌ {time_str} FAILED: {data.get('node_name')}")
        print(f"   {'':10} └── Error: {data.get('error')}")
        
    elif event_type == 'workflow_completed':
        print(f"\n🎉 {time_str} Workflow completed!")
        print(f"   {'':10} └── Status: {data.get('status')}")
        print(f"   {'':10} └── Nodes executed: {data.get('nodes_executed')}")
        print(f"   {'':10} └── Nodes failed: {data.get('nodes_failed')}")
        print(f"   {'':10} └── Order: {' → '.join(data.get('execution_order', []))}")
        
    elif event_type == 'workflow_failed':
        print(f"💥 {time_str} Workflow FAILED!")
        print(f"   {'':10} └── Error: {data.get('error')}")
        
    elif event_type == 'error':
        print(f"💥 {time_str} ERROR!")
        print(f"   {'':10} └── {data.get('error')}")

if __name__ == "__main__":
    demo_streaming_workflow()
