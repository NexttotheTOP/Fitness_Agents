import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { generateProfileOverview, queryFitnessCoach } from '../../api';
import { useToast } from "@/components/ui/use-toast";

export interface FitnessProfileData {
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

const ACTIVITY_LEVELS = [
    { value: "sedentary", label: "Sedentary (little or no exercise)" },
    { value: "lightly_active", label: "Lightly active (1-3 days/week)" },
    { value: "moderately_active", label: "Moderately active (3-5 days/week)" },
    { value: "very_active", label: "Very active (6-7 days/week)" },
    { value: "extra_active", label: "Extra active (very active & physical job)" },
];

export default function FitnessProfile() {
    const [markdown, setMarkdown] = useState('');
    const [loading, setLoading] = useState(false);
    const [threadId, setThreadId] = useState('');
    const [query, setQuery] = useState('');

    const handleSubmit = async (event: React.FormEvent) => {
        event.preventDefault();
        setLoading(true);
        setMarkdown('');

        const formData = new FormData(event.target as HTMLFormElement);
        const profileData: FitnessProfileData = {
            age: parseInt(formData.get('age') as string),
            gender: formData.get('gender') as string,
            height: formData.get('height') as string,
            weight: formData.get('weight') as string,
            activity_level: formData.get('activity_level') as string,
            fitness_goals: (formData.getAll('fitness_goals') as string[]),
            dietary_preferences: (formData.getAll('dietary_preferences') as string[]),
            health_restrictions: (formData.getAll('health_restrictions') as string[]),
        };

        try {
            const eventSource = await generateProfileOverview(profileData, (updatedMarkdown) => {
                setMarkdown(updatedMarkdown);
            });

            eventSource.addEventListener('complete', () => {
                setLoading(false);
                eventSource.close();
            });

            // Store the thread ID for future queries
            setThreadId(profileData.thread_id || '');
        } catch (error) {
            console.error('Error creating profile:', error);
            setLoading(false);
        }
    };

    const handleQuery = async (event: React.FormEvent) => {
        event.preventDefault();
        if (!threadId || !query.trim()) return;

        setLoading(true);
        setMarkdown('');

        try {
            const eventSource = await queryFitnessCoach(threadId, query, (updatedMarkdown) => {
                setMarkdown(updatedMarkdown);
            });

            eventSource.addEventListener('complete', () => {
                setLoading(false);
                eventSource.close();
            });
        } catch (error) {
            console.error('Error querying fitness coach:', error);
            setLoading(false);
        }
    };

    return (
        <div className="container mx-auto px-4 py-8">
            <h1 className="text-3xl font-bold mb-8">Fitness Profile Creator</h1>
            
            {/* Profile Creation Form */}
            <form onSubmit={handleSubmit} className="space-y-6 mb-8">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <label className="block text-sm font-medium text-gray-700">Age</label>
                        <input
                            type="number"
                            name="age"
                            required
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                        />
                    </div>
                    
                    <div>
                        <label className="block text-sm font-medium text-gray-700">Gender</label>
                        <select
                            name="gender"
                            required
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                        >
                            <option value="male">Male</option>
                            <option value="female">Female</option>
                            <option value="other">Other</option>
                        </select>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700">Height</label>
                        <input
                            type="text"
                            name="height"
                            placeholder="e.g., 5'10&quot; or 178cm"
                            required
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700">Weight</label>
                        <input
                            type="text"
                            name="weight"
                            placeholder="e.g., 150lbs or 68kg"
                            required
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700">Activity Level</label>
                        <select
                            name="activity_level"
                            required
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                        >
                            {ACTIVITY_LEVELS.map((level) => (
                                <option key={level.value} value={level.value}>
                                    {level.label}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700">Fitness Goals</label>
                    <div className="mt-2 space-y-2">
                        {['weight_loss', 'muscle_gain', 'endurance', 'strength', 'flexibility'].map((goal) => (
                            <div key={goal} className="flex items-center">
                                <input
                                    type="checkbox"
                                    name="fitness_goals"
                                    value={goal}
                                    className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                                />
                                <label className="ml-2 text-sm text-gray-700">
                                    {goal.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                                </label>
                            </div>
                        ))}
                    </div>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700">Dietary Preferences</label>
                    <div className="mt-2 space-y-2">
                        {['vegetarian', 'vegan', 'pescatarian', 'keto', 'paleo', 'mediterranean'].map((pref) => (
                            <div key={pref} className="flex items-center">
                                <input
                                    type="checkbox"
                                    name="dietary_preferences"
                                    value={pref}
                                    className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                                />
                                <label className="ml-2 text-sm text-gray-700">
                                    {pref.charAt(0).toUpperCase() + pref.slice(1)}
                                </label>
                            </div>
                        ))}
                    </div>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700">Health Restrictions</label>
                    <div className="mt-2 space-y-2">
                        {['gluten_free', 'dairy_free', 'nut_allergy', 'diabetes', 'heart_condition'].map((restriction) => (
                            <div key={restriction} className="flex items-center">
                                <input
                                    type="checkbox"
                                    name="health_restrictions"
                                    value={restriction}
                                    className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                                />
                                <label className="ml-2 text-sm text-gray-700">
                                    {restriction.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                                </label>
                            </div>
                        ))}
                    </div>
                </div>

                <button
                    type="submit"
                    disabled={loading}
                    className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-indigo-400"
                >
                    {loading ? 'Creating Profile...' : 'Create Profile'}
                </button>
            </form>

            {/* Query Form */}
            {threadId && (
                <form onSubmit={handleQuery} className="space-y-4 mb-8">
                    <div>
                        <label className="block text-sm font-medium text-gray-700">Ask your fitness coach</label>
                        <textarea
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                            rows={3}
                            placeholder="Ask about your diet, workout plan, or any fitness-related questions..."
                        />
                    </div>
                    <button
                        type="submit"
                        disabled={loading || !query.trim()}
                        className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-indigo-400"
                    >
                        {loading ? 'Processing...' : 'Send Query'}
                    </button>
                </form>
            )}

            {/* Response Display */}
            {markdown && (
                <div className="prose prose-lg max-w-none bg-white p-6 rounded-lg shadow-lg">
                    <ReactMarkdown
                        components={{
                            // Ensure proper spacing for lists
                            ul: ({children, ...props}) => (
                                <ul className="space-y-2" {...props}>{children}</ul>
                            ),
                            li: ({children, ...props}) => (
                                <li className="leading-relaxed" {...props}>{children}</li>
                            ),
                            // Ensure proper heading spacing
                            h1: ({children, ...props}) => (
                                <h1 className="text-3xl font-bold mt-8 mb-4" {...props}>{children}</h1>
                            ),
                            h2: ({children, ...props}) => (
                                <h2 className="text-2xl font-semibold mt-6 mb-3" {...props}>{children}</h2>
                            ),
                            // Ensure proper paragraph spacing
                            p: ({children, ...props}) => (
                                <p className="my-4 leading-relaxed" {...props}>{children}</p>
                            ),
                            // Style code blocks and inline code
                            code: ({children, className, ...props}) => {
                                const isInline = !className;
                                return isInline ? (
                                    <code className="px-1 py-0.5 bg-gray-100 rounded" {...props}>{children}</code>
                                ) : (
                                    <code className="block bg-gray-100 p-4 rounded" {...props}>{children}</code>
                                );
                            },
                        }}
                    >
                        {markdown}
                    </ReactMarkdown>
                </div>
            )}
        </div>
    );
} 