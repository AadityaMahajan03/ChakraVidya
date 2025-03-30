import React from 'react';
import { Users, BookOpen, Play, MessageCircle, Award, Zap, Rocket, Star, Trophy } from "lucide-react";

const BalVidyaCommunitySocial = () => {
    const features = [
        {
            icon: <MessageCircle className="text-white" size={24} />,
            title: "SPACE CHAT",
            description: "Connect with space cadets in your fleet",
            bgColor: "from-blue-600/40 to-blue-500/20",
            borderColor: "border-blue-300"
        },
        {
            icon: <BookOpen className="text-white" size={24} />,
            title: "KNOWLEDGE PORT",
            description: "Get help from galactic teachers",
            bgColor: "from-green-600/40 to-green-500/20",
            borderColor: "border-green-300"
        },
        {
            icon: <Zap className="text-white" size={24} />,
            title: "POWER QUIZZES",
            description: "Battle friends in cosmic knowledge wars",
            bgColor: "from-purple-600/40 to-purple-500/20",
            borderColor: "border-purple-300"
        },
        {
            icon: <Trophy className="text-white" size={24} />,
            title: "TROPHY VAULT",
            description: "Show off your stellar achievements",
            bgColor: "from-yellow-600/40 to-yellow-500/20",
            borderColor: "border-yellow-300"
        },
        {
            icon: <Users className="text-white" size={24} />,
            title: "CREW MEMBERS",
            description: "Find your space squad online now",
            bgColor: "from-pink-600/40 to-pink-500/20",
            borderColor: "border-pink-300"
        },
        {
            icon: <Rocket className="text-white" size={24} />,
            title: "LIVE MISSIONS",
            description: "Join special space challenges today",
            bgColor: "from-indigo-600/40 to-indigo-500/20",
            borderColor: "border-indigo-300"
        }
    ];

    return (
        <div className="bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-400 p-6 min-h-screen">
            <div className="max-w-5xl mx-auto">
                {/* Fun Animated Header */}
                <div className="bg-indigo-800/80 backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 border-yellow-300 mb-6 text-white relative overflow-hidden">
                    <div className="absolute -right-8 -top-8 w-32 h-32 bg-yellow-300 rounded-full opacity-30 animate-pulse"></div>
                    <div className="absolute right-20 top-10 w-16 h-16 bg-pink-300 rounded-full opacity-40 animate-bounce"></div>

                    <div className="flex items-center justify-between">
                        <div>
                            <h1 className="text-3xl font-bold mb-1 flex items-center">
                                <Rocket className="mr-3 text-yellow-300 animate-bounce" size={32} />
                                COSMIC COMMUNITY
                            </h1>
                            <p className="text-blue-200 text-xl">Find your space crew! 🌟</p>
                        </div>
                        <div className="bg-indigo-700/80 p-3 rounded-xl flex items-center border-2 border-blue-300">
                            <div className="mr-3">
                                <Zap className="text-yellow-300" size={28} />
                            </div>
                            <div>
                                <p className="text-base text-blue-200 font-bold">ONLINE FRIENDS</p>
                                <div className="w-36 h-4 bg-indigo-900 rounded-full mt-1 border border-blue-400">
                                    <div className="w-2/3 h-full bg-gradient-to-r from-yellow-300 to-orange-400 rounded-full animate-pulse"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    {features.map((feature, index) => (
                        <div
                            key={index}
                            className={`bg-gradient-to-r ${feature.bgColor} backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 ${feature.borderColor} transform hover:scale-105 transition-all`}
                        >
                            <div className="flex items-center mb-3">
                                <div className={`p-3 rounded-xl mr-3 border-2 border-white bg-${feature.borderColor.split('-')[1]}-500`}>
                                    {feature.icon}
                                </div>
                                <h3 className="font-bold text-xl text-white">{feature.title}</h3>
                            </div>
                            <p className="text-blue-100 font-bold ml-2 mb-3">{feature.description}</p>

                            <div className="bg-white/30 p-3 rounded-xl flex items-center justify-between border-2 border-white">
                                <div className="flex -space-x-2">
                                    <div className="w-8 h-8 rounded-full bg-blue-500 border-2 border-white flex items-center justify-center">
                                        <span className="text-white text-xs font-bold">S</span>
                                    </div>
                                    <div className="w-8 h-8 rounded-full bg-green-500 border-2 border-white flex items-center justify-center">
                                        <span className="text-white text-xs font-bold">A</span>
                                    </div>
                                    <div className="w-8 h-8 rounded-full bg-purple-500 border-2 border-white flex items-center justify-center">
                                        <span className="text-white text-xs font-bold">R</span>
                                    </div>
                                </div>
                                <span className={`text-base bg-indigo-800 text-white px-3 py-1 rounded-xl font-bold border-2 border-white ${index % 2 === 0 ? 'animate-pulse' : ''}`}>
                                    {index % 2 === 0 ? 'NEW!' : 'JOIN!'}
                                </span>
                            </div>
                        </div>
                    ))}
                </div>

                <div className="bg-pink-600/40 backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 border-pink-300">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-xl font-bold text-white flex items-center">
                            <MessageCircle className="text-pink-300 mr-2 animate-bounce" size={28} />
                            PREVIOUS CHATS
                        </h3>
                        <div className="bg-pink-800 text-white px-3 py-1 rounded-xl flex items-center font-bold border-2 border-white">
                            <Star className="text-yellow-300 mr-1" size={20} />
                            <span className="text-base">5 CHATS</span>
                        </div>
                    </div>
                    <div className="space-y-3">
                        <div className="bg-gradient-to-r from-blue-500/40 to-blue-400/20 p-4 rounded-xl border-4 border-blue-400 flex items-center">
                            <Users className="text-blue-300 mr-3" size={28} />
                            <div>
                                <p className="text-white font-bold text-lg">Space Math Group</p>
                                <div className="flex items-center">
                                    <div className="bg-blue-800 px-2 py-1 rounded-lg mr-2 text-base font-bold text-white border border-white">3 MESSAGES</div>
                                    <p className="text-white font-bold">Today</p>
                                </div>
                            </div>
                        </div>
                        <div className="bg-gradient-to-r from-purple-500/40 to-purple-400/20 p-4 rounded-xl border-4 border-purple-400 flex items-center">
                            <BookOpen className="text-purple-300 mr-3" size={28} />
                            <div>
                                <p className="text-white font-bold text-lg">Homework Help</p>
                                <div className="flex items-center">
                                    <div className="bg-purple-800 px-2 py-1 rounded-lg mr-2 text-base font-bold text-white border border-white">7 MESSAGES</div>
                                    <p className="text-white font-bold">Yesterday</p>
                                </div>
                            </div>
                        </div>
                        <div className="flex justify-center mt-4">
                            <button className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-6 py-2 rounded-xl text-lg font-bold border-2 border-white hover:from-indigo-700 hover:to-purple-700 transition-all transform hover:scale-105">
                                VIEW ALL CHATS
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default BalVidyaCommunitySocial;