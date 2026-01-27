---
sidebar_position: 12
---

# Troubleshooting and Debugging Agentic AI Systems

Agentic AI systems present unique debugging challenges due to their complex, dynamic, and autonomous nature. This chapter provides comprehensive strategies for identifying, diagnosing, and resolving issues in agentic systems.

## Understanding Agentic System Failures

### Types of Agentic System Failures

#### Planning Failures
Planning failures occur when agents fail to create effective strategies to achieve goals:

- **Incomplete Planning**: Missing critical steps in the plan
- **Invalid Planning**: Plans containing impossible or invalid actions
- **Suboptimal Planning**: Plans that are highly inefficient
- **Inconsistent Planning**: Plans that contradict themselves or context

```python
class PlanningFailureAnalyzer:
    def __init__(self):
        self.failure_patterns = {
            'incomplete': self.analyze_incomplete_planning,
            'invalid': self.analyze_invalid_planning,
            'suboptimal': self.analyze_suboptimal_planning,
            'inconsistent': self.analyze_inconsistent_planning
        }
    
    def analyze_plan_failure(self, failed_plan, goal, context):
        """Analyze why a plan failed"""
        failure_analysis = {}
        
        for failure_type, analyzer in self.failure_patterns.items():
            analysis = analyzer(failed_plan, goal, context)
            if analysis['detected']:
                failure_analysis[failure_type] = analysis
        
        return {
            'failure_types': list(failure_analysis.keys()),
            'detailed_analysis': failure_analysis,
            'recommendations': self.generate_recommendations(failure_analysis)
        }
    
    def analyze_incomplete_planning(self, plan, goal, context):
        """Analyze if plan is missing critical steps"""
        # Check if plan addresses all aspects of the goal
        goal_decomposition = self.decompose_goal(goal)
        plan_coverage = self.assess_plan_coverage(plan, goal_decomposition)
        
        return {
            'detected': plan_coverage < 0.9,  # Less than 90% coverage
            'coverage_ratio': plan_coverage,
            'missing_elements': self.identify_missing_elements(plan, goal_decomposition)
        }
```

#### Reasoning Failures
Issues in logical reasoning and decision-making processes:

- **Logical Inconsistencies**: Contradictory reasoning chains
- **Confirmation Bias**: Over-weighting information that confirms initial beliefs
- **Availability Bias**: Over-relying on immediately available information
- **Sunk Cost Fallacy**: Continuing with failing strategies due to invested resources

```python
class ReasoningFailureDetector:
    def __init__(self):
        self.bias_detectors = {
            'confirmation': self.detect_confirmation_bias,
            'availability': self.detect_availability_bias,
            'sunk_cost': self.detect_sunk_cost_fallacy
        }
    
    def detect_reasoning_failures(self, reasoning_trace):
        """Detect various types of reasoning failures"""
        bias_analysis = {}
        
        for bias_type, detector in self.bias_detectors.items():
            bias_analysis[bias_type] = detector(reasoning_trace)
        
        return {
            'detected_biases': [bias for bias, result in bias_analysis.items() if result['detected']],
            'bias_severity': self.assess_bias_severity(bias_analysis),
            'mitigation_suggestions': self.generate_mitigation_suggestions(bias_analysis)
        }
    
    def detect_confirmation_bias(self, reasoning_trace):
        """Detect confirmation bias in reasoning process"""
        evidence_types = self.categorize_evidence(reasoning_trace)
        
        # Check if agent primarily sought evidence supporting initial hypothesis
        supporting_evidence_ratio = evidence_types['supporting'] / max(
            evidence_types['total'], 1
        )
        
        # Look for ignored contradictory evidence
        ignored_contradictory = evidence_types['ignored_contradictory']
        
        severe_bias = (
            supporting_evidence_ratio > 0.8 and 
            ignored_contradictory > 0.3 * evidence_types['total']
        )
        
        return {
            'detected': severe_bias,
            'supporting_ratio': supporting_evidence_ratio,
            'ignored_contradictory_count': ignored_contradictory
        }
```

#### Memory and Context Failures
Issues related to information management and context preservation:

- **Context Drift**: Losing track of original goal or context
- **Memory Overload**: Information overload affecting performance
- **Selective Memory**: Forgetting crucial information
- **False Memory**: Storing incorrect or misleading information

```python
class MemoryFailureAnalyzer:
    def __init__(self):
        self.context_tracker = ContextTracker()
        self.memory_assessor = MemoryAssessmentSystem()
    
    def analyze_memory_failure(self, agent_state, expected_context, actual_output):
        """Analyze memory-related failures"""
        # Check for context drift
        context_drift_analysis = self.analyze_context_drift(
            agent_state, expected_context
        )
        
        # Assess memory quality
        memory_quality = self.memory_assessor.evaluate_memory_quality(agent_state)
        
        # Check for information consistency
        consistency_analysis = self.check_information_consistency(agent_state)
        
        return {
            'context_drift': context_drift_analysis,
            'memory_quality': memory_quality,
            'consistency_issues': consistency_analysis,
            'memory_failure_type': self.classify_memory_failure(
                context_drift_analysis, memory_quality, consistency_analysis
            )
        }
    
    def analyze_context_drift(self, agent_state, expected_context):
        """Analyze if agent has drifted from original context"""
        current_context = self.context_tracker.extract_current_context(agent_state)
        
        # Calculate similarity between expected and actual context
        context_similarity = self.calculate_context_similarity(
            expected_context, current_context
        )
        
        # Check if key context elements are preserved
        preserved_elements = self.check_context_preservation(
            expected_context, current_context
        )
        
        drift_detected = context_similarity < 0.7  # 70% similarity threshold
        
        return {
            'drift_detected': drift_detected,
            'similarity_score': context_similarity,
            'preserved_elements': preserved_elements,
            'drift_severity': self.assess_drift_severity(context_similarity)
        }
```

## Debugging Strategies

### Trace-Based Debugging

#### Step-by-Step Execution Tracing
```python
class AgentExecutionTracer:
    def __init__(self):
        self.trace_log = []
        self.watch_points = []
        self.exception_handler = ExceptionHandler()
    
    def trace_execution(self, agent, goal, context):
        """Trace agent execution step by step"""
        self.trace_log = []  # Reset trace log
        
        try:
            # Initialize tracing
            initial_state = {
                'goal': goal,
                'context': context,
                'timestamp': time.time(),
                'step': 0,
                'action': 'initialization'
            }
            self.trace_log.append(initial_state)
            
            # Execute with tracing
            result = self.execute_with_tracing(agent, goal, context)
            
            return {
                'result': result,
                'execution_trace': self.trace_log,
                'success': True
            }
        except Exception as e:
            # Log exception in trace
            exception_entry = {
                'step': len(self.trace_log),
                'action': 'exception',
                'exception': str(e),
                'traceback': traceback.format_exc(),
                'state_at_failure': self.get_current_state(agent)
            }
            self.trace_log.append(exception_entry)
            
            return {
                'result': None,
                'execution_trace': self.trace_log,
                'success': False,
                'error': str(e)
            }
    
    def execute_with_tracing(self, agent, goal, context):
        """Execute agent with detailed tracing"""
        step_count = 0
        
        # Agent loop with tracing
        while not agent.is_goal_achieved(goal):
            # Observe step
            observation = agent.observe(context)
            self.log_trace_step(step_count, 'observe', observation)
            
            # Think step
            thought = agent.think(observation, goal)
            self.log_trace_step(step_count, 'think', thought)
            
            # Plan step
            plan = agent.plan(thought, goal)
            self.log_trace_step(step_count, 'plan', plan)
            
            # Act step
            action_result = agent.act(plan)
            self.log_trace_step(step_count, 'act', action_result)
            
            step_count += 1
            
            # Check for execution limits
            if step_count > agent.max_steps:
                raise Exception(f"Agent exceeded maximum steps ({agent.max_steps})")
        
        return agent.get_final_result()
    
    def log_trace_step(self, step_num, action_type, data):
        """Log a step in the execution trace"""
        trace_entry = {
            'step': step_num,
            'action_type': action_type,
            'data': data,
            'timestamp': time.time(),
            'memory_snapshot': self.capture_memory_state(),
            'context_snapshot': self.capture_context_state()
        }
        self.trace_log.append(trace_entry)
```

#### Interactive Debugging
```python
class InteractiveAgentDebugger:
    def __init__(self):
        self.breakpoint_manager = BreakpointManager()
        self.state_inspector = StateInspector()
        self.variable_watcher = VariableWatcher()
    
    def debug_agent_interactive(self, agent, goal, context):
        """Debug agent execution interactively"""
        print("Starting interactive debugging session...")
        print(f"Goal: {goal}")
        print(f"Initial context: {context}")
        
        step_count = 0
        breakpoints_hit = 0
        
        while not agent.is_goal_achieved(goal):
            # Check for breakpoints
            current_state = agent.get_internal_state()
            breakpoint_condition = self.check_breakpoints(current_state, step_count)
            
            if breakpoint_condition:
                print(f"Breakpoint hit at step {step_count}")
                print(f"Current state: {self.state_inspector.inspect(current_state)}")
                
                # Interactive debugging session
                user_command = self.get_user_debug_command()
                
                if user_command == 'continue':
                    pass  # Continue execution
                elif user_command == 'inspect':
                    self.inspect_state_interactive(current_state)
                elif user_command == 'modify':
                    new_state = self.modify_state_interactive(current_state)
                    agent.set_internal_state(new_state)
                elif user_command == 'step':
                    step_count += 1
                    continue
                elif user_command == 'quit':
                    break
            
            # Execute next step
            agent.step()
            step_count += 1
            
            if step_count > agent.max_steps:
                print(f"Max steps reached: {agent.max_steps}")
                break
    
    def inspect_state_interactive(self, state):
        """Interactive state inspection"""
        while True:
            print("Available variables to inspect:")
            for i, var in enumerate(state.keys()):
                print(f"{i}: {var}")
            
            try:
                selection = input("Enter variable name or index (or 'done' to continue): ")
                
                if selection == 'done':
                    break
                elif selection.isdigit():
                    var_name = list(state.keys())[int(selection)]
                    print(f"{var_name}: {state[var_name]}")
                else:
                    print(f"{selection}: {state.get(selection, 'Not found')}")
            except KeyboardInterrupt:
                break
```

### Log-Based Debugging

#### Comprehensive Logging System
```python
class AgenticAILogger:
    def __init__(self, log_level='INFO', log_format='json'):
        self.logger = logging.getLogger('agentic_ai')
        self.log_level = log_level
        self.log_format = log_format
        self.log_history = []
        
        # Set up logging
        self.setup_logging()
    
    def setup_logging(self):
        """Set up comprehensive logging system"""
        handler = logging.StreamHandler()
        
        if self.log_format == 'json':
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(getattr(logging, self.log_level))
    
    def log_agent_decision(self, agent_id, decision, reasoning, context):
        """Log agent decision-making process"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'agent_id': agent_id,
            'event_type': 'agent_decision',
            'decision': decision,
            'reasoning': reasoning,
            'context_summary': self.summarize_context(context),
            'session_id': context.get('session_id')
        }
        
        self.logger.info(json.dumps(log_entry))
        self.log_history.append(log_entry)
    
    def log_tool_execution(self, agent_id, tool_name, parameters, result, success):
        """Log tool execution results"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'agent_id': agent_id,
            'event_type': 'tool_execution',
            'tool_name': tool_name,
            'parameters': self.sanitize_parameters(parameters),
            'result_summary': self.summarize_result(result),
            'success': success,
            'execution_time': parameters.get('execution_time')
        }
        
        self.logger.info(json.dumps(log_entry))
        self.log_history.append(log_entry)
    
    def summarize_context(self, context):
        """Create a safe summary of context"""
        return {
            'task_type': context.get('task_type'),
            'complexity': context.get('complexity'),
            'required_tools': context.get('required_tools', [])[:3]  # First 3 tools
        }
    
    def sanitize_parameters(self, parameters):
        """Remove sensitive information from parameters"""
        if isinstance(parameters, dict):
            sanitized = parameters.copy()
            # Remove sensitive keys
            sensitive_keys = ['password', 'token', 'key', 'secret']
            for key in sensitive_keys:
                if key in sanitized:
                    sanitized[key] = '[REDACTED]'
            return sanitized
        return parameters
```

## Common Debugging Techniques

### State Inspection

#### Memory State Analysis
```python
class MemoryStateAnalyzer:
    def __init__(self):
        self.memory_comparator = MemoryComparator()
        self.relevance_scoring = RelevanceScoringSystem()
        self.compression_analyzer = CompressionAnalyzer()
    
    def analyze_memory_state(self, memory_content, query_context):
        """Analyze the state of agent memory"""
        # Assess memory relevance
        relevance_analysis = self.assess_memory_relevance(memory_content, query_context)
        
        # Check for memory quality
        quality_metrics = self.assess_memory_quality(memory_content)
        
        # Analyze compression effectiveness
        compression_analysis = self.analyze_compression(memory_content)
        
        # Identify potential memory conflicts
        conflict_analysis = self.identify_memory_conflicts(memory_content)
        
        return {
            'relevance_score': relevance_analysis['score'],
            'relevance_issues': relevance_analysis['issues'],
            'quality_metrics': quality_metrics,
            'compression_analysis': compression_analysis,
            'conflict_analysis': conflict_analysis,
            'recommended_actions': self.generate_memory_actions(
                relevance_analysis, quality_metrics, compression_analysis, conflict_analysis
            )
        }
    
    def assess_memory_relevance(self, memory_content, query_context):
        """Assess how relevant memory is to current context"""
        relevance_scores = []
        irrelevant_items = []
        
        for item in memory_content:
            relevance = self.relevance_scoring.calculate(item, query_context)
            relevance_scores.append(relevance)
            
            if relevance < 0.3:  # Threshold for irrelevance
                irrelevant_items.append(item)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        return {
            'score': avg_relevance,
            'issues': {
                'high_irrelevance_count': len(irrelevant_items),
                'irrelevant_items': irrelevant_items
            }
        }
```

#### Context Window Analysis
```python
class ContextWindowAnalyzer:
    def __init__(self):
        self.token_analyzer = TokenAnalyzer()
        self.context_relevance = ContextRelevanceSystem()
        self.attention_analysis = AttentionAnalysisSystem()
    
    def analyze_context_window(self, context_window, current_task):
        """Analyze the context window for potential issues"""
        # Analyze token distribution
        token_analysis = self.token_analyzer.analyze(context_window)
        
        # Assess context relevance
        relevance_analysis = self.context_relevance.assess(
            context_window, current_task
        )
        
        # Analyze attention patterns
        attention_analysis = self.attention_analysis.analyze(
            context_window, current_task
        )
        
        # Identify potential context issues
        issues = self.identify_context_issues(
            token_analysis, relevance_analysis, attention_analysis
        )
        
        return {
            'token_analysis': token_analysis,
            'relevance_analysis': relevance_analysis,
            'attention_analysis': attention_analysis,
            'identified_issues': issues,
            'optimization_recommendations': self.generate_context_optimization(
                issues, token_analysis, relevance_analysis
            )
        }
    
    def identify_context_issues(self, token_analysis, relevance_analysis, attention_analysis):
        """Identify common context-related issues"""
        issues = []
        
        # Check for token limit issues
        if token_analysis['token_count'] > 0.95 * token_analysis['max_tokens']:
            issues.append({
                'type': 'token_limit_approaching',
                'severity': 'high',
                'details': f"Context at {token_analysis['token_percentage']:.1%} of limit"
            })
        
        # Check for relevance issues
        if relevance_analysis['average_relevance'] < 0.4:
            issues.append({
                'type': 'low_relevance_context',
                'severity': 'medium',
                'details': f"Average relevance: {relevance_analysis['average_relevance']:.2f}"
            })
        
        # Check for attention issues
        if attention_analysis['focused_attention_ratio'] < 0.2:
            issues.append({
                'type': 'poor_attention_focus',
                'severity': 'medium',
                'details': f"Only {attention_analysis['focused_attention_ratio']:.1%} of context receiving focused attention"
            })
        
        return issues
```

### Performance Profiling

#### Execution Profiling
```python
class ExecutionProfiler:
    def __init__(self):
        self.timer = ExecutionTimer()
        self.resource_monitor = ResourceMonitor()
        self.bottleneck_detector = BottleneckDetector()
    
    def profile_agent_execution(self, agent, task, context):
        """Profile agent execution for performance issues"""
        profile_data = {
            'execution_phases': {},
            'resource_usage': {},
            'bottlenecks': [],
            'optimization_opportunities': []
        }
        
        # Profile each phase of execution
        with self.timer.time_phase('initialization'):
            agent.initialize(context)
        
        with self.timer.time_phase('planning'):
            plan = agent.plan(task, context)
        
        with self.timer.time_phase('execution'):
            result = agent.execute(plan)
        
        with self.timer.time_phase('cleanup'):
            agent.cleanup()
        
        # Monitor resource usage throughout
        profile_data['execution_phases'] = self.timer.get_phase_times()
        profile_data['resource_usage'] = self.resource_monitor.get_usage_summary()
        
        # Identify bottlenecks
        profile_data['bottlenecks'] = self.bottleneck_detector.identify_bottlenecks(
            profile_data['execution_phases']
        )
        
        # Generate optimization recommendations
        profile_data['optimization_opportunities'] = self.generate_optimization_recommendations(
            profile_data
        )
        
        return profile_data
    
    def generate_optimization_recommendations(self, profile_data):
        """Generate optimization recommendations based on profiling"""
        recommendations = []
        
        # Identify slow phases
        slow_phases = [
            (phase, time) for phase, time in profile_data['execution_phases'].items()
            if time > 5.0  # More than 5 seconds
        ]
        
        for phase, time in slow_phases:
            recommendations.append({
                'phase': phase,
                'time_spent': time,
                'recommendation': f'Optimize {phase} phase - taking {time:.2f}s',
                'priority': 'high' if time > 10.0 else 'medium'
            })
        
        # Identify resource bottlenecks
        for resource_type, usage in profile_data['resource_usage'].items():
            if usage['utilization'] > 0.8:  # Over 80% utilization
                recommendations.append({
                    'resource': resource_type,
                    'utilization': usage['utilization'],
                    'recommendation': f'Optimize {resource_type} usage - currently {usage["utilization"]:.1%}',
                    'priority': 'high'
                })
        
        return recommendations
```

## Failure Recovery Strategies

### Error Classification and Recovery

#### Systematic Error Handling
```python
class AgenticErrorRecoverySystem:
    def __init__(self):
        self.error_classifiers = {
            'planning_error': PlanningErrorClassifier(),
            'execution_error': ExecutionErrorClassifier(),
            'tool_error': ToolErrorClassifier(),
            'memory_error': MemoryErrorClassifier(),
            'communication_error': CommunicationErrorClassifier()
        }
        self.recovery_strategies = RecoveryStrategyLibrary()
    
    def handle_agent_error(self, error, agent_state, context):
        """Handle errors in agentic systems with appropriate recovery"""
        # Classify the error
        error_type = self.classify_error(error, agent_state, context)
        
        # Get appropriate recovery strategy
        recovery_strategy = self.recovery_strategies.get_strategy(error_type)
        
        # Attempt recovery
        recovery_result = recovery_strategy.execute(
            error, agent_state, context
        )
        
        if recovery_result['success']:
            # Recovery successful
            return {
                'recovery_successful': True,
                'new_agent_state': recovery_result['new_state'],
                'recovery_strategy_used': recovery_result['strategy_used'],
                'actions_taken': recovery_result['actions']
            }
        else:
            # Recovery failed, may need escalation
            return {
                'recovery_successful': False,
                'error': error,
                'recovery_attempts': recovery_result['attempts'],
                'escalation_needed': True
            }
    
    def classify_error(self, error, agent_state, context):
        """Classify error type for appropriate handling"""
        for error_type, classifier in self.error_classifiers.items():
            if classifier.matches(error, agent_state, context):
                return error_type
        
        # Default to general error if no specific type matches
        return 'general_error'
```

#### Adaptive Recovery Mechanisms
```python
class AdaptiveRecoverySystem:
    def __init__(self):
        self.recovery_memory = RecoveryMemory()
        self.effectiveness_tracker = EffectivenessTracker()
        self.strategy_selector = StrategySelector()
    
    def adapt_recovery_approach(self, error_history, current_error):
        """Adapt recovery approach based on past effectiveness"""
        # Analyze similar past errors
        similar_past_errors = self.recovery_memory.find_similar_errors(current_error)
        
        # Assess which strategies worked for similar errors
        effective_strategies = self.effectiveness_tracker.get_effective_strategies(
            similar_past_errors
        )
        
        # Select best strategy for current situation
        selected_strategy = self.strategy_selector.select_best_strategy(
            current_error, effective_strategies
        )
        
        # Apply recovery with learning
        recovery_result = selected_strategy.execute(current_error)
        
        # Learn from outcome
        self.recovery_memory.add_recovery_attempt(
            current_error, selected_strategy, recovery_result
        )
        
        self.effectiveness_tracker.update_effectiveness(
            selected_strategy, recovery_result['success']
        )
        
        return recovery_result
```

## Testing and Validation

### Unit Testing for Agentic Components

#### Component-Specific Testing
```python
import unittest
from unittest.mock import Mock, patch

class TestAgentComponents(unittest.TestCase):
    def setUp(self):
        self.agent = Agent()
        self.mock_llm = Mock()
        self.mock_memory = Mock()
        self.mock_tools = Mock()
        
        # Inject mocks
        self.agent.llm = self.mock_llm
        self.agent.memory = self.mock_memory
        self.agent.tools = self.mock_tools
    
    def test_planning_component(self):
        """Test planning component with various inputs"""
        test_cases = [
            {
                'goal': 'research quantum computing',
                'context': {'knowledge_domain': 'physics'},
                'expected_steps': 5
            },
            {
                'goal': 'send email',
                'context': {'user_preferences': {'email_format': 'formal'}},
                'expected_steps': 2
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                plan = self.agent.plan(test_case['goal'], test_case['context'])
                
                self.assertIsNotNone(plan)
                self.assertGreater(len(plan), 0)
                self.assertLessEqual(len(plan), test_case['expected_steps'])
    
    def test_memory_retrieval(self):
        """Test memory retrieval functionality"""
        # Setup memory with test data
        test_memory = [
            {'type': 'fact', 'content': 'Paris is the capital of France', 'timestamp': time.time()},
            {'type': 'event', 'content': 'Meeting scheduled for 2 PM', 'timestamp': time.time() - 3600}
        ]
        self.mock_memory.retrieve.return_value = test_memory
        
        # Test retrieval
        retrieved = self.agent.memory.retrieve_relevant('France')
        
        self.assertEqual(len(retrieved), 1)
        self.assertIn('France', retrieved[0]['content'])
    
    def test_tool_integration(self):
        """Test tool integration and execution"""
        # Mock tool response
        self.mock_tools.execute.return_value = {'result': 'success', 'data': 'test data'}
        
        # Test tool execution
        result = self.agent.execute_tool('search', {'query': 'test'})
        
        self.mock_tools.execute.assert_called_once_with('search', {'query': 'test'})
        self.assertEqual(result['result'], 'success')
```

#### Integration Testing
```python
class AgentIntegrationTester:
    def __init__(self, agent):
        self.agent = agent
        self.test_scenarios = self.load_test_scenarios()
    
    def run_integration_tests(self):
        """Run comprehensive integration tests"""
        results = {
            'passed': 0,
            'failed': 0,
            'errors': [],
            'test_results': []
        }
        
        for scenario in self.test_scenarios:
            try:
                test_result = self.execute_test_scenario(scenario)
                results['test_results'].append(test_result)
                
                if test_result['status'] == 'PASS':
                    results['passed'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(test_result['error'])
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(str(e))
        
        return results
    
    def execute_test_scenario(self, scenario):
        """Execute a specific test scenario"""
        try:
            # Setup initial state
            self.setup_scenario_context(scenario)
            
            # Execute agent
            result = self.agent.run(scenario['input'])
            
            # Validate result
            validation = self.validate_scenario_result(scenario, result)
            
            return {
                'scenario': scenario['name'],
                'status': 'PASS' if validation['passed'] else 'FAIL',
                'details': validation['details'],
                'result': result,
                'error': None
            }
        except Exception as e:
            return {
                'scenario': scenario['name'],
                'status': 'ERROR',
                'details': str(e),
                'result': None,
                'error': str(e)
            }
    
    def load_test_scenarios(self):
        """Load test scenarios from test specification"""
        return [
            {
                'name': 'simple_fact_lookup',
                'input': 'What is the capital of France?',
                'expected_outputs': ['Paris', 'country', 'capital'],
                'timeout': 30
            },
            {
                'name': 'multi_step_reasoning',
                'input': 'Plan a 3-day trip to Tokyo',
                'expected_outputs': ['transportation', 'accommodation', 'activities'],
                'timeout': 120
            },
            {
                'name': 'tool_integration',
                'input': 'Search for current weather in New York',
                'expected_outputs': ['temperature', 'weather'],
                'timeout': 60
            }
        ]
```

## Monitoring and Alerting

### Real-Time Monitoring

#### Performance Monitoring
```python
class AgenticSystemMonitor:
    def __init__(self, alert_thresholds):
        self.metrics_collector = MetricsCollector()
        self.alert_system = AlertSystem()
        self.thresholds = alert_thresholds
        self.performance_history = []
    
    def monitor_agent_performance(self, agent_id, metrics):
        """Monitor agent performance metrics"""
        # Collect current metrics
        current_metrics = self.metrics_collector.collect(agent_id, metrics)
        
        # Check against thresholds
        alerts = []
        for metric_name, metric_value in current_metrics.items():
            threshold = self.thresholds.get(metric_name, float('inf'))
            
            if metric_value > threshold:
                alert = self.alert_system.create_alert(
                    agent_id, metric_name, metric_value, threshold
                )
                alerts.append(alert)
        
        # Store performance history
        self.performance_history.append({
            'timestamp': time.time(),
            'agent_id': agent_id,
            'metrics': current_metrics,
            'alerts_generated': len(alerts)
        })
        
        return {
            'metrics': current_metrics,
            'alerts': alerts,
            'performance_trend': self.analyze_trend(agent_id)
        }
    
    def analyze_trend(self, agent_id):
        """Analyze performance trends over time"""
        agent_history = [entry for entry in self.performance_history 
                        if entry['agent_id'] == agent_id]
        
        if len(agent_history) < 2:
            return 'insufficient_data'
        
        # Calculate trend for key metrics
        response_time_trend = self.calculate_trend(
            [entry['metrics'].get('response_time', 0) for entry in agent_history]
        )
        
        success_rate_trend = self.calculate_trend(
            [entry['metrics'].get('success_rate', 1.0) for entry in agent_history]
        )
        
        return {
            'response_time_trend': response_time_trend,
            'success_rate_trend': success_rate_trend
        }
    
    def calculate_trend(self, values):
        """Calculate trend direction from a list of values"""
        if len(values) < 2:
            return 'neutral'
        
        # Simple linear trend
        first_value = values[0]
        last_value = values[-1]
        
        if last_value > first_value * 1.1:  # 10% increase
            return 'increasing'
        elif last_value < first_value * 0.9:  # 10% decrease
            return 'decreasing'
        else:
            return 'stable'
```

#### Health Checks
```python
class AgentHealthChecker:
    def __init__(self):
        self.health_indicators = {
            'memory_health': self.check_memory_health,
            'planning_health': self.check_planning_health,
            'tool_health': self.check_tool_health,
            'communication_health': self.check_communication_health
        }
    
    def perform_health_check(self, agent):
        """Perform comprehensive health check on agent"""
        health_report = {}
        
        for indicator, checker in self.health_indicators.items():
            health_report[indicator] = checker(agent)
        
        overall_health = self.calculate_overall_health(health_report)
        
        return {
            'health_report': health_report,
            'overall_health_score': overall_health,
            'health_status': self.health_score_to_status(overall_health),
            'recommended_actions': self.get_health_recommendations(health_report)
        }
    
    def check_memory_health(self, agent):
        """Check memory system health"""
        memory_stats = agent.memory.get_statistics()
        
        # Check for memory leaks
        memory_growth_rate = self.calculate_memory_growth_rate(memory_stats)
        
        # Check for fragmentation
        fragmentation_score = self.calculate_fragmentation_score(memory_stats)
        
        # Check for relevance degradation
        relevance_score = self.calculate_relevance_score(agent.memory)
        
        health_score = 1.0 - (
            min(1.0, memory_growth_rate * 0.3) +
            min(1.0, fragmentation_score * 0.3) +
            min(1.0, (1 - relevance_score) * 0.4)
        )
        
        return {
            'health_score': health_score,
            'memory_growth_rate': memory_growth_rate,
            'fragmentation_score': fragmentation_score,
            'relevance_score': relevance_score,
            'issues': self.identify_memory_issues(
                memory_growth_rate, fragmentation_score, relevance_score
            )
        }
    
    def check_planning_health(self, agent):
        """Check planning system health"""
        # Get recent planning statistics
        planning_stats = agent.get_planning_statistics()
        
        # Calculate planning efficiency
        avg_plan_length = sum(planning_stats['plan_lengths']) / len(planning_stats['plan_lengths'])
        success_rate = planning_stats['successful_plans'] / len(planning_stats['plan_lengths'])
        
        # Check for planning consistency
        consistency_score = self.calculate_planning_consistency(planning_stats)
        
        health_score = (
            min(1.0, success_rate) * 0.5 +
            min(1.0, 10 / avg_plan_length) * 0.3 +  # Prefer shorter plans
            consistency_score * 0.2
        )
        
        return {
            'health_score': health_score,
            'avg_plan_length': avg_plan_length,
            'success_rate': success_rate,
            'consistency_score': consistency_score,
            'issues': self.identify_planning_issues(
                avg_plan_length, success_rate, consistency_score
            )
        }
```

## Debugging Tools and Utilities

### Diagnostic Utilities

#### Agent State Inspector
```python
class AgentStateInspector:
    def __init__(self):
        self.state_schema = self.define_state_schema()
    
    def inspect_agent_state(self, agent):
        """Comprehensive inspection of agent internal state"""
        state = agent.get_internal_state()
        
        inspection_report = {
            'memory_state': self.inspect_memory_state(state.get('memory')),
            'planning_state': self.inspect_planning_state(state.get('planning')),
            'tool_state': self.inspect_tool_state(state.get('tools')),
            'context_state': self.inspect_context_state(state.get('context')),
            'execution_state': self.inspect_execution_state(state.get('execution')),
            'configuration_state': self.inspect_configuration_state(state.get('config'))
        }
        
        # Overall assessment
        inspection_report['overall_assessment'] = self.assess_overall_state(inspection_report)
        
        return inspection_report
    
    def inspect_memory_state(self, memory_state):
        """Inspect memory state for potential issues"""
        if not memory_state:
            return {'status': 'unavailable', 'issues': ['Memory state not available']}
        
        issues = []
        
        # Check memory size
        if isinstance(memory_state, dict) and 'entries' in memory_state:
            memory_size = len(memory_state['entries'])
            if memory_size > 10000:  # Threshold for concern
                issues.append(f'Large memory size: {memory_size} entries')
        
        # Check memory freshness
        if 'last_access' in memory_state:
            time_since_access = time.time() - memory_state['last_access']
            if time_since_access > 3600:  # 1 hour
                issues.append('Memory not accessed recently')
        
        return {
            'status': 'healthy' if not issues else 'issues_found',
            'size': memory_size if 'memory_size' in locals() else 0,
            'issues': issues
        }
    
    def assess_overall_state(self, inspection_report):
        """Assess overall agent health based on inspection"""
        critical_issues = []
        warning_issues = []
        
        for component, report in inspection_report.items():
            if isinstance(report, dict) and 'issues' in report:
                for issue in report['issues']:
                    if 'critical' in issue.lower():
                        critical_issues.append(f"{component}: {issue}")
                    elif 'high' in issue.lower() or 'severe' in issue.lower():
                        critical_issues.append(f"{component}: {issue}")
                    else:
                        warning_issues.append(f"{component}: {issue}")
        
        if critical_issues:
            return {'status': 'critical', 'issues': critical_issues}
        elif warning_issues:
            return {'status': 'warnings', 'issues': warning_issues}
        else:
            return {'status': 'healthy', 'issues': []}
```

#### Performance Profiler
```python
class AgentPerformanceProfiler:
    def __init__(self):
        self.baseline_metrics = self.load_baseline_metrics()
    
    def profile_agent(self, agent, test_workload):
        """Profile agent performance under test workload"""
        start_time = time.time()
        
        # Run test workload
        results = []
        for task in test_workload:
            task_start = time.time()
            result = agent.process(task)
            task_time = time.time() - task_start
            
            results.append({
                'task': task,
                'result': result,
                'processing_time': task_time,
                'success': self.check_success(result)
            })
        
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        metrics = {
            'total_processing_time': total_time,
            'average_task_time': sum(r['processing_time'] for r in results) / len(results),
            'success_rate': sum(1 for r in results if r['success']) / len(results),
            'throughput': len(results) / total_time
        }
        
        # Compare with baseline
        comparison = self.compare_with_baseline(metrics)
        
        return {
            'performance_metrics': metrics,
            'baseline_comparison': comparison,
            'task_results': results,
            'recommendations': self.generate_performance_recommendations(comparison)
        }
    
    def generate_performance_recommendations(self, comparison):
        """Generate recommendations based on performance comparison"""
        recommendations = []
        
        for metric, comparison_data in comparison.items():
            if comparison_data['ratio'] > 1.5:  # Performance is 50% worse than baseline
                recommendations.append({
                    'metric': metric,
                    'issue': f"Performance {metric} is {comparison_data['ratio']:.2f}x worse than baseline",
                    'recommendation': self.get_specific_recommendation(metric),
                    'severity': 'high'
                })
            elif comparison_data['ratio'] > 1.2:  # Performance is 20% worse than baseline
                recommendations.append({
                    'metric': metric,
                    'issue': f"Performance {metric} is {comparison_data['ratio']:.2f}x worse than baseline",
                    'recommendation': self.get_specific_recommendation(metric),
                    'severity': 'medium'
                })
        
        return recommendations
    
    def get_specific_recommendation(self, metric):
        """Get specific recommendation for a metric"""
        recommendations = {
            'average_task_time': "Consider optimizing the planning or reasoning components",
            'success_rate': "Review error handling and fallback strategies",
            'throughput': "Investigate potential bottlenecks in execution pipeline"
        }
        return recommendations.get(metric, "Review general performance optimization strategies")
```

## Best Practices for Troubleshooting

### Systematic Debugging Approach

When troubleshooting agentic AI systems, follow this systematic approach:

1. **Reproduce the Issue**: Ensure you can consistently reproduce the problem
2. **Isolate the Component**: Identify which component is failing
3. **Check Inputs**: Verify the inputs to the system are correct
4. **Trace Execution**: Follow the execution path to identify where things go wrong
5. **Analyze State**: Examine the internal state when the issue occurs
6. **Test Hypothesis**: Formulate and test hypotheses about the root cause
7. **Implement Fix**: Apply the appropriate fix
8. **Verify Resolution**: Confirm the issue is resolved and no new issues are introduced

### Documentation and Knowledge Sharing

Maintain comprehensive documentation of:

- Common failure patterns and their solutions
- Debugging procedures for different types of issues
- Performance benchmarks and expected behavior
- Recovery procedures for different failure modes

## Conclusion

Effective troubleshooting of agentic AI systems requires:

1. **Deep Understanding**: Understanding the complex interactions between planning, reasoning, memory, and execution components
2. **Comprehensive Tools**: Having the right tools for monitoring, logging, and debugging
3. **Systematic Approach**: Following systematic procedures to identify and resolve issues
4. **Proactive Monitoring**: Implementing monitoring to catch issues early
5. **Continuous Learning**: Learning from failures to improve system robustness

As agentic systems become more sophisticated, debugging and troubleshooting methodologies will need to evolve accordingly. The key is to maintain visibility into the system's decision-making process while respecting privacy and security requirements.

The future of agentic AI debugging will likely involve more sophisticated AI-assisted debugging tools that can analyze complex execution traces and suggest solutions automatically, making it easier to maintain and improve these complex systems.