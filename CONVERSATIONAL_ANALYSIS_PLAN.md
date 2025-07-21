# TTBall Conversational Analysis Planning

## 🎯 Overview

This document outlines the implementation of a **conversational analysis mode** where users can "chat" with the AI about analyzed videos, leveraging both visual content and database insights to provide rich, contextual responses.

## 🧠 Core Concept: Context-Aware AI Conversations

### The Vision
Users can ask natural language questions about their table tennis videos and get intelligent responses that combine:
- **Visual analysis** from Gemma 3N multimodal AI
- **Database insights** from stored trajectory and anomaly data  
- **Historical context** from previous analyses
- **Physics knowledge** about table tennis gameplay

### Example Conversations
```
User: "What happened around the 32-second mark in this video?"
AI: "At 32 seconds, I detected a high-speed rally with the ball reaching 67 km/h. The ball bounced on the far side of the table with a 15° spin angle. Based on the trajectory data, this was an aggressive topspin shot that created an unusual bounce pattern - the ball accelerated by 12% after the bounce, which is above normal physics expectations."

User: "Were there any anomalies in the bouncing patterns?"
AI: "Yes, I found 3 bounce anomalies in this video: 1) At 32s - energy gain during bounce (impossible physics), 2) At 1:15 - ball trajectory break suggesting tracking loss, 3) At 2:03 - confidence drop from 0.9 to 0.3 during paddle contact. The 32s anomaly is particularly interesting as it might indicate spin effects not captured in the standard physics model."

User: "Show me the most exciting rally"
AI: "The most exciting rally was from 1:45-2:12 (27 seconds). It featured 18 ball exchanges, speeds ranging 45-78 km/h, and 3 edge shots. The rally ended with an unusual bounce at 2:12 where the ball changed direction by 23° more than physics would predict - possibly due to advanced spin technique."
```

## 🔧 Technical Architecture

### New Analysis Type: "chat"
```javascript
// Frontend interface
analysis_type: "chat"
chat_options: {
    include_visual_analysis: true,
    include_database_context: true,  
    include_anomaly_data: true,
    max_context_frames: 10,
    conversation_history: []
}
```

### Backend Processing Flow
```python
# New endpoint: /analyze/chat
@app.post("/analyze/chat")
async def analyze_with_chat_interface(
    file: UploadFile = File(...),
    user_question: str = Form(...),
    conversation_history: List[Dict] = Form([]),
    include_context: bool = Form(True)
):
    # Step 1: Analyze video if not already done
    video_analysis = get_or_create_video_analysis(file)
    
    # Step 2: Gather relevant context
    context = build_conversation_context(
        video_analysis=video_analysis,
        database_insights=get_database_insights(video_analysis.video_id),
        user_question=user_question,
        conversation_history=conversation_history
    )
    
    # Step 3: Generate contextual response
    response = gemma_conversational_analysis(
        question=user_question,
        context=context,
        visual_frames=get_relevant_frames(video_analysis, user_question)
    )
    
    return {
        "response": response,
        "context_used": context,
        "relevant_timestamps": extract_timestamps_from_response(response),
        "suggested_followups": generate_followup_questions(response)
    }
```

## 📊 Context Building System

### 1. Video Analysis Context
```python
def build_video_context(video_analysis):
    """Build context from video analysis results"""
    return {
        "video_metadata": {
            "duration": video_analysis.duration,
            "fps": video_analysis.fps,
            "resolution": video_analysis.resolution
        },
        "detection_summary": {
            "total_detections": video_analysis.ball_detections,
            "detection_rate": video_analysis.detection_rate,
            "avg_confidence": video_analysis.avg_confidence
        },
        "trajectory_info": {
            "trajectory_points": video_analysis.trajectory_points,
            "speed_analysis": extract_speed_stats(video_analysis),
            "bounce_events": count_bounce_events(video_analysis)
        }
    }
```

### 2. Database Insights Context
```python
def build_database_context(video_id):
    """Build context from database insights"""
    # Get detection data
    detections = get_ball_detections(video_id)
    
    # Get anomaly data if available
    anomalies = get_anomaly_analysis(video_id)
    
    # Get bounce events if available
    bounces = get_bounce_events(video_id)
    
    # Calculate insights
    context = {
        "key_moments": identify_key_moments(detections),
        "speed_patterns": analyze_speed_patterns(detections),
        "anomaly_summary": summarize_anomalies(anomalies),
        "bounce_analysis": summarize_bounces(bounces),
        "rally_structure": analyze_rally_structure(detections)
    }
    
    return context
```

### 3. Question-Specific Frame Selection
```python
def get_relevant_frames(video_analysis, user_question):
    """Select frames most relevant to user question"""
    question_keywords = extract_keywords(user_question)
    
    # Time-based questions
    if any(time_word in question_keywords for time_word in ["when", "second", "minute", "time"]):
        timestamps = extract_timestamps_from_question(user_question)
        return get_frames_at_timestamps(video_analysis, timestamps)
    
    # Event-based questions  
    elif any(event_word in question_keywords for event_word in ["bounce", "rally", "serve", "shot"]):
        return get_frames_for_events(video_analysis, question_keywords)
    
    # Anomaly questions
    elif any(anomaly_word in question_keywords for anomaly_word in ["anomaly", "unusual", "strange", "weird"]):
        return get_frames_with_anomalies(video_analysis)
    
    # Default: key moments
    else:
        return get_key_moment_frames(video_analysis)
```

## 🤖 Enhanced Gemma 3N Integration

### Conversational Prompts
```python
def create_conversational_prompt(question, context, frames_info):
    """Create rich conversational prompt for Gemma 3N"""
    
    prompt = f"""You are an expert table tennis analyst with access to detailed video analysis data. 
A user is asking about their table tennis video, and you have comprehensive information to provide insights.

USER QUESTION: {question}

VIDEO CONTEXT:
- Duration: {context['video_metadata']['duration']}s
- Ball detections: {context['detection_summary']['total_detections']} 
- Detection rate: {context['detection_summary']['detection_rate']:.1%}
- Average speed: {context.get('avg_speed', 'N/A')} km/h

DATABASE INSIGHTS:
- Key moments: {context.get('key_moments', [])}
- Anomalies detected: {len(context.get('anomaly_summary', []))}
- Bounce events: {len(context.get('bounce_analysis', []))}
- Rally count: {context.get('rally_count', 'Unknown')}

VISUAL FRAMES ANALYZED:
{format_frames_info(frames_info)}

INSTRUCTIONS:
1. Answer the user's question specifically and accurately
2. Use the database insights to provide detailed information
3. Reference specific timestamps when relevant
4. Explain any anomalies or unusual patterns found
5. Provide actionable insights for table tennis improvement
6. Be conversational but informative

RESPONSE:"""
    
    return prompt
```

### Multi-Modal Analysis Enhancement
```python
def gemma_conversational_analysis(question, context, visual_frames):
    """Enhanced Gemma analysis for conversational interface"""
    
    # Initialize Gemma system
    gemma = FixedGemmaMultimodal()
    
    # Process each relevant frame
    frame_analyses = []
    for frame_info in visual_frames:
        frame_analysis = gemma.analyze_frame_multimodal(
            frame=frame_info['frame'],
            prompt_type="conversational",
            additional_context=f"User asked: {question}. Frame timestamp: {frame_info['timestamp']}s"
        )
        frame_analyses.append(frame_analysis)
    
    # Create comprehensive response
    conversational_prompt = create_conversational_prompt(question, context, frame_analyses)
    
    # Generate response
    response = gemma.generate_conversational_response(
        prompt=conversational_prompt,
        max_tokens=300,  # Longer responses for conversations
        temperature=0.4  # Slightly more creative for natural conversation
    )
    
    # Post-process response
    processed_response = post_process_conversational_response(response, context)
    
    return processed_response
```

## 📱 Frontend Chat Interface

### Chat UI Components
```javascript
// Chat interface component
class VideoChat {
    constructor(videoId, apiUrl) {
        this.videoId = videoId;
        this.apiUrl = apiUrl;
        this.conversationHistory = [];
        this.contextData = null;
    }
    
    async initializeChat() {
        // Load video context
        this.contextData = await this.loadVideoContext();
        
        // Show initial suggestions
        this.showSuggestedQuestions();
    }
    
    async askQuestion(question) {
        // Add user message to chat
        this.addMessage('user', question);
        
        // Show typing indicator
        this.showTyping();
        
        try {
            // Send question to AI
            const response = await fetch(`${this.apiUrl}/analyze/chat`, {
                method: 'POST',
                body: new FormData({
                    video_id: this.videoId,
                    user_question: question,
                    conversation_history: JSON.stringify(this.conversationHistory),
                    include_context: true
                })
            });
            
            const result = await response.json();
            
            // Add AI response to chat
            this.addMessage('ai', result.response);
            
            // Show suggested follow-ups
            if (result.suggested_followups) {
                this.showFollowupSuggestions(result.suggested_followups);
            }
            
            // Update conversation history
            this.conversationHistory.push({
                question: question,
                response: result.response,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            this.addMessage('error', 'Sorry, I encountered an error processing your question.');
        }
        
        this.hideTyping();
    }
    
    showSuggestedQuestions() {
        const suggestions = [
            "What happened in the most exciting part of this video?",
            "Were there any unusual bouncing patterns?", 
            "How fast was the ball moving on average?",
            "Show me the key moments in this rally",
            "What anomalies did you detect?",
            "How does this video compare to typical gameplay?"
        ];
        
        this.displaySuggestions(suggestions);
    }
}
```

### Enhanced Analytics Dashboard
```javascript
// Add chat tab to existing analytics dashboard
const analyticsEnhancements = {
    addChatTab: function() {
        const chatTab = `
            <div class="tab-pane" id="chat">
                <div class="chat-container">
                    <div class="chat-header">
                        <h3>Ask about your video</h3>
                        <p>Chat with AI about gameplay, anomalies, and insights</p>
                    </div>
                    
                    <div class="chat-messages" id="chatMessages">
                        <!-- Messages will appear here -->
                    </div>
                    
                    <div class="chat-input">
                        <input type="text" id="chatInput" placeholder="Ask about your video..." />
                        <button id="sendChat">Send</button>
                    </div>
                    
                    <div class="chat-suggestions" id="chatSuggestions">
                        <!-- Suggested questions will appear here -->
                    </div>
                </div>
            </div>
        `;
        
        document.getElementById('analyticsContent').insertAdjacentHTML('beforeend', chatTab);
    }
};
```

## 🗄️ Database Integration

### Conversation History Storage
```sql
-- Store conversation history
CREATE TABLE conversation_history (
    id INTEGER PRIMARY KEY,
    video_id INTEGER,
    session_id VARCHAR,
    user_question TEXT,
    ai_response TEXT,
    context_used TEXT,  -- JSON of context data used
    relevant_timestamps TEXT,  -- JSON array of timestamps referenced
    created_at TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES video_metadata(id)
);

-- Store conversation insights
CREATE TABLE conversation_insights (
    id INTEGER PRIMARY KEY,
    video_id INTEGER,
    insight_type VARCHAR,  -- "key_moment", "anomaly_explanation", etc.
    insight_text TEXT,
    confidence FLOAT,
    source_question TEXT,  -- Original user question that generated this insight
    created_at TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES video_metadata(id)
);
```

### Enhanced Context Queries
```python
@app.get("/chat/context/{video_id}")
async def get_chat_context(video_id: int):
    """Get comprehensive context for chat about a video"""
    
    # Base video data
    video_data = get_video_metadata(video_id)
    detection_data = get_ball_detections(video_id)
    
    # Enhanced insights
    context = {
        "video_summary": summarize_video_performance(detection_data),
        "key_moments": identify_key_moments(detection_data),
        "speed_analysis": analyze_ball_speeds(detection_data),
        "rally_structure": analyze_rally_patterns(detection_data),
        "comparative_insights": compare_with_average_performance(detection_data)
    }
    
    # Add anomaly data if available
    if anomaly_data := get_anomaly_analysis(video_id):
        context["anomaly_insights"] = summarize_anomalies_for_chat(anomaly_data)
    
    # Add bounce data if available  
    if bounce_data := get_bounce_events(video_id):
        context["bounce_insights"] = summarize_bounces_for_chat(bounce_data)
    
    return context
```

## 🎯 Advanced Conversation Features

### 1. Follow-up Question Generation
```python
def generate_followup_questions(ai_response, context):
    """Generate relevant follow-up questions based on AI response"""
    
    followups = []
    
    # If response mentions anomalies
    if "anomaly" in ai_response.lower():
        followups.append("Can you explain why this anomaly occurred?")
        followups.append("How does this compare to normal gameplay?")
    
    # If response mentions specific timestamps
    if any(char.isdigit() for char in ai_response) and "second" in ai_response:
        followups.append("What happened right before this moment?")
        followups.append("Show me the trajectory at this time")
    
    # If response mentions speed or physics
    if any(word in ai_response.lower() for word in ["speed", "fast", "slow", "bounce"]):
        followups.append("How does this speed compare to professional play?")
        followups.append("What techniques could improve this?")
    
    # Generic useful followups
    followups.extend([
        "What can I learn from this analysis?",
        "Are there any patterns I should notice?",
        "How can I improve my technique based on this?"
    ])
    
    return followups[:4]  # Return top 4 suggestions
```

### 2. Natural Language Timestamp Extraction
```python
def extract_timestamps_from_question(question):
    """Extract timestamps from natural language questions"""
    import re
    
    # Look for patterns like "32 seconds", "1:30", "2 minutes"
    patterns = [
        r'(\d+)\s*seconds?',
        r'(\d+):(\d+)',  # mm:ss format
        r'(\d+)\s*minutes?\s*(\d+)?\s*seconds?'
    ]
    
    timestamps = []
    for pattern in patterns:
        matches = re.findall(pattern, question)
        for match in matches:
            # Convert to seconds
            if len(match) == 1:  # Just seconds
                timestamps.append(int(match[0]))
            elif len(match) == 2:  # mm:ss or minutes + seconds
                timestamps.append(int(match[0]) * 60 + int(match[1]))
    
    return timestamps
```

### 3. Intelligent Context Prioritization
```python
def prioritize_context_for_question(question, full_context):
    """Prioritize most relevant context based on question type"""
    
    question_lower = question.lower()
    prioritized_context = {}
    
    # Time-based questions - prioritize temporal data
    if any(word in question_lower for word in ["when", "time", "second", "minute"]):
        prioritized_context = {
            "timeline_events": full_context.get("key_moments", []),
            "temporal_anomalies": full_context.get("time_based_anomalies", [])
        }
    
    # Performance questions - prioritize statistics
    elif any(word in question_lower for word in ["fast", "speed", "performance", "good", "bad"]):
        prioritized_context = {
            "speed_analysis": full_context.get("speed_analysis", {}),
            "performance_metrics": full_context.get("performance_summary", {})
        }
    
    # Anomaly questions - prioritize anomaly data
    elif any(word in question_lower for word in ["anomaly", "unusual", "strange", "wrong"]):
        prioritized_context = {
            "anomaly_details": full_context.get("anomaly_insights", []),
            "physics_violations": full_context.get("physics_anomalies", [])
        }
    
    # Default: provide balanced context
    else:
        prioritized_context = {
            "summary": full_context.get("video_summary", {}),
            "highlights": full_context.get("key_moments", [])[:3],
            "quick_stats": full_context.get("quick_stats", {})
        }
    
    return prioritized_context
```

## 🚀 Implementation Phases

### Phase 1: Basic Chat Interface
1. **Add "chat" analysis type** to existing endpoints
2. **Create basic conversational prompts** for Gemma 3N  
3. **Build simple context from database** (detections, metadata)
4. **Implement basic chat UI** in dashboard

### Phase 2: Enhanced Context Integration
1. **Add anomaly data** to conversation context
2. **Implement intelligent frame selection** based on questions
3. **Create follow-up question generation**
4. **Add conversation history storage**

### Phase 3: Advanced Conversational Features
1. **Natural language timestamp parsing**
2. **Comparative analysis** ("how does this compare to...")
3. **Proactive insights** ("you might also be interested in...")
4. **Multi-turn conversation** memory and context

### Phase 4: Smart Assistance
1. **Coaching suggestions** based on analysis
2. **Pattern recognition** across multiple videos
3. **Predictive insights** ("based on this pattern...")
4. **Integration with external knowledge** (table tennis rules, techniques)

## 🎯 Expected User Experience

### Natural Conversations
```
User: "This video looks weird around 1:30"
AI: "I analyzed the 1:30 mark and found an interesting anomaly. The ball gained 15% energy during a table bounce, which violates physics laws. This suggests either a tracking error or an unusual spin effect. The ball speed increased from 52 km/h to 60 km/h during the bounce, which is impossible without external force."

User: "Is that a problem with my analysis?"  
AI: "It could be either a tracking issue or a real spin effect. Looking at the confidence scores, they dropped from 0.89 to 0.67 during this event, suggesting some uncertainty in detection. I recommend checking if there was unusual lighting or if the player used advanced spin technique."

User: "How can I tell the difference?"
AI: "Great question! Check these indicators: 1) If confidence scores are low (< 0.7), it's likely tracking error. 2) If the ball trajectory before/after looks smooth, it might be real spin. 3) Watch for paddle angle in the previous shot - extreme angles can create unusual spin. In your case, confidence was borderline, so review the visual at 1:28-1:32."
```

## 🏆 Success Metrics

### Technical Performance
- **Response Time**: < 3 seconds for most questions
- **Context Accuracy**: > 90% relevant information in responses  
- **Conversation Flow**: > 85% follow-up questions rated as relevant
- **Integration Success**: Seamless access to all database insights

### User Value
- **Engagement**: Users ask average 5+ questions per video
- **Insight Discovery**: Users discover insights they wouldn't find manually
- **Learning**: Users report improved understanding of their gameplay
- **Satisfaction**: > 90% find conversational interface helpful

---

## 📝 Conclusion

The conversational analysis mode represents a significant evolution in sports video analysis, transforming raw data into an interactive, intelligent assistant. By combining Gemma 3N's multimodal capabilities with comprehensive database insights, users get unprecedented access to their gameplay analysis through natural conversation.

**Key Innovation**: This approach makes complex sports analytics accessible to everyone, regardless of technical expertise, while providing deeper insights than traditional static reports. 