import React from "react";
import { Shield, Zap, Award, Gamepad2 } from "lucide-react";

const BalVidyaSafetyParentalControl = () => {
    return (
        <div className="w-full h-screen bg-gradient-to-br from-purple-600 via-indigo-500 to-blue-600 text-white p-6 overflow-auto">
            {/* Decorative Elements */}
            <div className="absolute top-10 right-10 w-32 h-32 bg-yellow-400 rounded-full opacity-60 blur-md"></div>
            <div className="absolute bottom-20 left-10 w-40 h-40 bg-pink-500 rounded-full opacity-40 blur-md"></div>
            <div className="absolute top-1/2 right-20 w-48 h-48 bg-blue-400 rounded-full opacity-30 blur-md"></div>
            <div className="absolute bottom-1/3 left-1/3 w-24 h-24 bg-green-400 rounded-full opacity-20 blur-md"></div>

            <div className="container mx-auto">
                <div className="flex items-center justify-between mb-8 pt-4">
                    <h2 className="text-3xl md:text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-yellow-300 to-green-300">Safety Command Center</h2>
                    <Gamepad2 className="text-yellow-300 h-10 w-10" />
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                    {/* Screen Time Tracker */}
                    <div className="border-2 border-cyan-400 rounded-xl p-6 bg-indigo-800 bg-opacity-40 backdrop-blur-sm hover:border-cyan-300 transition-all duration-300 transform hover:-translate-y-1 h-full">
                        <div className="flex items-center mb-6">
                            <div className="bg-cyan-500 p-3 rounded-lg mr-3">
                                <Zap className="text-yellow-300 h-6 w-6" />
                            </div>
                            <h3 className="font-bold text-xl text-cyan-300">SCREEN TIME POWERS</h3>
                        </div>
                        <div className="text-center py-4">
                            <div className="inline-flex items-center justify-center w-40 h-40 rounded-full bg-gradient-to-br from-indigo-900 to-blue-600 border-4 border-cyan-400 mb-6 relative">
                                <div className="absolute inset-2 rounded-full border-4 border-t-cyan-300 border-r-transparent border-b-transparent border-l-transparent animate-spin"></div>
                                <div className="text-center">
                                    <p className="text-4xl font-black text-cyan-300">45</p>
                                    <p className="text-lg font-medium">minutes today</p>
                                </div>
                            </div>
                            <div className="flex justify-center items-center">
                                <Award className="text-yellow-400 mr-2 h-6 w-6" />
                                <p className="text-cyan-100 text-lg font-medium">Daily Quest: 2 hours</p>
                            </div>
                        </div>
                    </div>

                    {/* Parental Lock */}
                    <div className="border-2 border-pink-400 rounded-xl p-6 bg-indigo-800 bg-opacity-40 backdrop-blur-sm hover:border-pink-300 transition-all duration-300 transform hover:-translate-y-1 h-full">
                        <div className="flex items-center mb-6">
                            <div className="bg-pink-600 p-3 rounded-lg mr-3">
                                <Shield className="text-yellow-300 h-6 w-6" />
                            </div>
                            <h3 className="font-bold text-xl text-pink-300">GUARDIAN SHIELDS</h3>
                        </div>
                        <div className="space-y-4 py-4">
                            {[
                                { label: "Content Force Field", status: "ACTIVE", icon: "✓", color: "green" },
                                { label: "Chat Radar System", status: "ACTIVE", icon: "✓", color: "green" },
                                { label: "Time Warp Controls", status: "ACTIVE", icon: "✓", color: "green" },
                            ].map((item, index) => (
                                <div key={index} className="flex items-center justify-between p-4 bg-gradient-to-r from-indigo-700 to-purple-700 bg-opacity-70 rounded-lg border border-pink-400 border-opacity-30">
                                    <span className="font-medium text-lg">{item.label}</span>
                                    <span className="bg-green-500 text-white font-bold px-4 py-2 rounded-full text-sm flex items-center">
                                        {item.status} {item.icon}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Additional Game-like Statistics Section */}
                <div className="border-2 border-yellow-400 rounded-xl p-6 bg-indigo-800 bg-opacity-40 backdrop-blur-sm mb-8">
                    <h3 className="font-bold text-xl text-yellow-300 mb-4 flex items-center">
                        <Award className="mr-2 h-6 w-6" />
                        COSMIC ACHIEVEMENTS
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="bg-gradient-to-r from-purple-700 to-indigo-700 p-4 rounded-lg text-center">
                            <p className="text-3xl font-bold text-green-300">7</p>
                            <p className="text-sm">Days Streak</p>
                        </div>
                        <div className="bg-gradient-to-r from-indigo-700 to-blue-700 p-4 rounded-lg text-center">
                            <p className="text-3xl font-bold text-cyan-300">15</p>
                            <p className="text-sm">Learning Missions</p>
                        </div>
                        <div className="bg-gradient-to-r from-blue-700 to-purple-700 p-4 rounded-lg text-center">
                            <p className="text-3xl font-bold text-pink-300">250</p>
                            <p className="text-sm">Star Points</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default BalVidyaSafetyParentalControl;