export interface FitnessProfileRequest {
    thread_id?: string;
    age: number;
    gender: string;
    height: string;
    weight: string;
    activity_level: string;
    fitness_goals: string[];
    dietary_preferences: string[];
    health_restrictions: string[];
    body_photos: string[]; // Not sure here if this is needed
}

export async function generateProfileOverview(
    data: FitnessProfileRequest,
    onMarkdownUpdate: (markdown: string) => void
): Promise<EventSource> {
    try {
        const response = await fetch('http://localhost:8000/fitness/profile', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Create an EventSource for the streaming response
        const eventSource = new EventSource(`http://localhost:8000/fitness/profile`);
        let accumulatedMarkdown = '';

        eventSource.onmessage = (event) => {
            try {
                // Remove the "data: " prefix and handle the content
                const content = event.data.replace(/^data: /, '').trim();
                
                // Skip empty messages
                if (!content) return;
                
                try {
                    // Try to parse as JSON
                    const jsonData = JSON.parse(content);
                    if (jsonData.content) {
                        accumulatedMarkdown += jsonData.content;
                    }
                } catch {
                    // If not JSON, treat as markdown
                    // Clean up any broken markdown formatting
                    const cleanedContent = content
                        .replace(/\s+/g, ' ')  // Replace multiple spaces with single space
                        .replace(/\s+([,\.])/g, '$1')  // Remove spaces before punctuation
                        .replace(/(\*\*)\s+/g, '**')  // Clean up bold markdown
                        .replace(/\s+(\*\*)/g, '**')
                        .replace(/(-)\s+/g, '-')  // Clean up list items
                        .trim();
                    
                    accumulatedMarkdown += cleanedContent + '\n';
                }
                
                onMarkdownUpdate(accumulatedMarkdown);
            } catch (error) {
                console.error('Error handling event data:', error);
            }
        };

        eventSource.onerror = (error) => {
            console.error('EventSource error:', error);
            eventSource.close();
        };

        return eventSource;
    } catch (error) {
        console.error('Error creating fitness profile:', error);
        throw error;
    }
}

export async function queryFitnessCoach(
    threadId: string,
    query: string,
    onMarkdownUpdate: (markdown: string) => void
): Promise<EventSource> {
    try {
        const response = await fetch('http://localhost:8000/fitness/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
            },
            body: JSON.stringify({ thread_id: threadId, query })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const eventSource = new EventSource(`http://localhost:8000/fitness/query?thread_id=${threadId}`);
        let accumulatedMarkdown = '';

        eventSource.onmessage = (event) => {
            try {
                // Remove the "data: " prefix and handle the content
                const content = event.data.replace(/^data: /, '').trim();
                
                // Skip empty messages
                if (!content) return;
                
                try {
                    // Try to parse as JSON
                    const jsonData = JSON.parse(content);
                    if (jsonData.content) {
                        accumulatedMarkdown += jsonData.content;
                    }
                } catch {
                    // If not JSON, treat as markdown
                    // Clean up any broken markdown formatting
                    const cleanedContent = content
                        .replace(/\s+/g, ' ')  // Replace multiple spaces with single space
                        .replace(/\s+([,\.])/g, '$1')  // Remove spaces before punctuation
                        .replace(/(\*\*)\s+/g, '**')  // Clean up bold markdown
                        .replace(/\s+(\*\*)/g, '**')
                        .replace(/(-)\s+/g, '-')  // Clean up list items
                        .trim();
                    
                    accumulatedMarkdown += cleanedContent + '\n';
                }
                
                onMarkdownUpdate(accumulatedMarkdown);
            } catch (error) {
                console.error('Error handling query response:', error);
            }
        };

        eventSource.onerror = (error) => {
            console.error('EventSource error:', error);
            eventSource.close();
        };

        return eventSource;
    } catch (error) {
        console.error('Error querying fitness coach:', error);
        throw error;
    }
}

// TypeScript interface for the profile data
interface FitnessProfileData {
    thread_id?: string;
    age: number;
    gender: string;
    height: string;
    weight: string;
    activity_level: string;
    fitness_goals: string[];
    dietary_preferences: string[];
    health_restrictions: string[];
}

// Example React component usage:
/*
import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';

const ProfileComponent = () => {
    const [markdown, setMarkdown] = useState('');
    
    const handleSubmit = async (profileData: FitnessProfileData) => {
        try {
            const eventSource = await createFitnessProfile(profileData, (updatedMarkdown) => {
                setMarkdown(updatedMarkdown);
            });
            
            // Clean up EventSource on component unmount
            return () => {
                eventSource.close();
            };
        } catch (error) {
            console.error('Error creating profile:', error);
        }
    };
    
    return (
        <div className="prose max-w-none">
            <ReactMarkdown>{markdown}</ReactMarkdown>
        </div>
    );
};
*/ 