import { Bot, BookOpen, Microscope, Play, Sparkles, Rocket, Brain, Zap } from "lucide-react";
import { useState } from "react";

const BalVidyaAILearningAssistant = () => {
    const [question, setQuestion] = useState("");
    const [recentQuestions, setRecentQuestions] = useState([
        "How does photosynthesis work?",
        "What is the difference between similes and metaphors?",
        "Help me with long division"
    ]);

    const handleAskQuestion = () => {
        if (question.trim()) {
            setRecentQuestions([question, ...recentQuestions.slice(0, 2)]);
            setQuestion("");
        }
    };

    return (
        <div className="w-full h-screen bg-gradient-to-br from-indigo-600 via-purple-600 to-blue-600 text-white p-6 overflow-auto">
            {/* Decorative Elements */}
            <div className="absolute top-20 right-20 w-32 h-32 bg-yellow-400 rounded-full opacity-50 blur-md"></div>
            <div className="absolute bottom-40 left-20 w-40 h-40 bg-pink-500 rounded-full opacity-30 blur-md"></div>
            <div className="absolute top-1/2 right-40 w-36 h-36 bg-cyan-400 rounded-full opacity-30 blur-md"></div>

            <div className="container mx-auto relative">
                <div className="flex items-center justify-between mb-8 pt-4">
                    <h2 className="text-3xl md:text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-300 to-pink-300">AI Learning Command Center</h2>
                    <Rocket className="text-yellow-300 h-10 w-10" />
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                    <div className="border-2 border-purple-400 rounded-xl p-6 bg-indigo-800 bg-opacity-40 backdrop-blur-sm hover:border-purple-300 transition-all duration-300 h-full">
                        <div className="flex items-center mb-6">
                            <div className="bg-purple-600 p-3 rounded-lg mr-3">
                                <Bot className="text-yellow-300 h-6 w-6" />
                            </div>
                            <h3 className="font-bold text-xl text-purple-300">QUANTUM BRAIN AI TUTOR</h3>
                        </div>
                        <div className="bg-purple-900 bg-opacity-50 rounded-xl p-4 mb-6 border border-purple-400">
                            <p className="mb-3 text-lg font-medium text-cyan-100">Ask me anything about your lessons!</p>
                            <div className="flex">
                                <input
                                    type="text"
                                    className="flex-1 border-2 border-purple-500 bg-indigo-900 bg-opacity-60 text-white rounded-l-lg p-3 placeholder-indigo-200"
                                    placeholder="Type your question..."
                                    value={question}
                                    onChange={(e) => setQuestion(e.target.value)}
                                />
                                <button
                                    className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white px-6 py-3 rounded-r-lg font-bold flex items-center"
                                    onClick={handleAskQuestion}
                                >
                                    <Zap className="mr-2 h-5 w-5" />
                                    ASK
                                </button>
                            </div>
                        </div>
                        <div className="border-t border-purple-400 pt-4">
                            <div className="flex items-center mb-3">
                                <Sparkles className="text-pink-300 mr-2 h-5 w-5" />
                                <p className="text-pink-200 font-medium">RECENT BRAIN QUESTS:</p>
                            </div>
                            <ul className="space-y-2 mt-2">
                                {recentQuestions.map((q, index) => (
                                    <li key={index} className="bg-indigo-700 bg-opacity-40 p-3 rounded-lg border border-purple-400 border-opacity-40 flex items-center">
                                        <div className="bg-purple-500 h-2 w-2 rounded-full mr-3"></div>
                                        <span>{q}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>

                    <div className="border-2 border-blue-400 rounded-xl p-6 bg-indigo-800 bg-opacity-40 backdrop-blur-sm hover:border-blue-300 transition-all duration-300 h-full">
                        <div className="flex items-center mb-6">
                            <div className="bg-blue-600 p-3 rounded-lg mr-3">
                                <Brain className="text-yellow-300 h-6 w-6" />
                            </div>
                            <h3 className="font-bold text-xl text-blue-300">STELLAR LEARNING MISSIONS</h3>
                        </div>
                        <p className="text-cyan-100 mb-5 font-medium">Based on your cosmic learning patterns, we recommend:</p>
                        <ul className="space-y-4">
                            <li className="flex p-4 bg-gradient-to-r from-blue-800 to-indigo-800 bg-opacity-60 rounded-lg border border-blue-400 border-opacity-40 hover:translate-x-1 transition-transform duration-200">
                                <div className="bg-blue-600 p-2 rounded-lg mr-3 flex-shrink-0">
                                    <BookOpen className="text-yellow-300 h-5 w-5" />
                                </div>
                                <div>
                                    <p className="font-medium text-blue-200">Math: Fractions Deep Dive</p>
                                    <p className="text-sm text-blue-100 opacity-80">You've been doing well, but this will help strengthen your skills</p>
                                </div>
                            </li>
                            <li className="flex p-4 bg-gradient-to-r from-green-800 to-blue-800 bg-opacity-60 rounded-lg border border-green-400 border-opacity-40 hover:translate-x-1 transition-transform duration-200">
                                <div className="bg-green-600 p-2 rounded-lg mr-3 flex-shrink-0">
                                    <Microscope className="text-yellow-300 h-5 w-5" />
                                </div>
                                <div>
                                    <p className="font-medium text-green-200">Science: Interactive Solar System</p>
                                    <p className="text-sm text-green-100 opacity-80">Learn about planets in our solar system</p>
                                </div>
                            </li>
                            <li className="flex p-4 bg-gradient-to-r from-yellow-800 to-orange-800 bg-opacity-60 rounded-lg border border-yellow-400 border-opacity-40 hover:translate-x-1 transition-transform duration-200">
                                <div className="bg-yellow-600 p-2 rounded-lg mr-3 flex-shrink-0">
                                    <Play className="text-indigo-900 h-5 w-5" />
                                </div>
                                <div>
                                    <p className="font-medium text-yellow-200">Grammar Games: Parts of Speech</p>
                                    <p className="text-sm text-yellow-100 opacity-80">Fun way to practice grammar rules</p>
                                </div>
                            </li>
                        </ul>
                        <div className="mt-6 flex justify-center">
                            <button className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-400 hover:to-purple-400 px-6 py-3 rounded-full font-bold text-white shadow-lg flex items-center">
                                <Rocket className="mr-2 h-5 w-5" />
                                LAUNCH NEW MISSION
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default BalVidyaAILearningAssistant;