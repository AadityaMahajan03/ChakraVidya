import React from 'react';
import { Glasses, Play, Zap, Rocket, BookOpen, Star, Target } from "lucide-react";

const BalVidyaAugmentedRealityLearning = () => {
    const arLessons = [
        { name: "Solar System", icon: "🪐", color: "from-blue-600 to-indigo-600" },
        { name: "Human Body", icon: "🫀", color: "from-red-500 to-pink-600" },
        { name: "Dinosaurs", icon: "🦖", color: "from-green-600 to-teal-600" },
        { name: "Geometry", icon: "📐", color: "from-purple-600 to-indigo-600" },
    ];

    return (
        <div className="bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-400 p-4 min-h-screen">
            <div className="max-w-5xl mx-auto">
                {/* Fun Animated Header */}
                <div className="bg-indigo-800/80 backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 border-yellow-300 mb-6 text-white relative overflow-hidden">
                    <div className="absolute -right-8 -top-8 w-32 h-32 bg-yellow-300 rounded-full opacity-30 animate-pulse"></div>
                    <div className="absolute right-20 top-10 w-16 h-16 bg-pink-300 rounded-full opacity-40 animate-bounce"></div>

                    <div className="flex items-center justify-between">
                        <div>
                            <h1 className="text-3xl font-bold mb-1 flex items-center">
                                <Glasses className="mr-3 text-yellow-300 animate-bounce" size={32} />
                                AR ADVENTURES!
                            </h1>
                            <p className="text-blue-200 text-xl">Explore amazing 3D worlds! 🚀</p>
                        </div>
                        <div className="bg-indigo-700/80 p-3 rounded-xl flex items-center border-2 border-blue-300">
                            <div className="mr-3">
                                <Zap className="text-yellow-300" size={28} />
                            </div>
                            <div>
                                <p className="text-base text-blue-200 font-bold">DISCOVERY POWER</p>
                                <div className="w-36 h-4 bg-indigo-900 rounded-full mt-1 border border-blue-400">
                                    <div className="w-3/4 h-full bg-gradient-to-r from-yellow-300 to-orange-400 rounded-full animate-pulse"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* AR Lessons Grid */}
                <div className="bg-blue-600/40 backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 border-blue-300 mb-6">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center text-white">
                            <div className="bg-blue-500 p-3 rounded-xl mr-3 border-2 border-white">
                                <Glasses size={24} className="text-white" />
                            </div>
                            <h3 className="font-bold text-xl">AR MAGIC LESSONS</h3>
                        </div>
                        <div className="bg-green-500 text-white px-3 py-2 rounded-xl text-base font-bold border-2 border-white">NEW WORLDS!</div>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {arLessons.map((lesson, index) => (
                            <div key={index} className="bg-white/30 backdrop-blur-md p-4 rounded-xl border-4 border-blue-300 hover:border-yellow-300 transform hover:scale-105 transition-all text-center cursor-pointer">
                                <div className={`bg-gradient-to-br ${lesson.color} w-20 h-20 mx-auto rounded-full flex items-center justify-center mb-3 border-4 border-white`}>
                                    <span className="text-3xl">{lesson.icon}</span>
                                </div>
                                <h4 className="text-white font-bold text-lg">{lesson.name}</h4>
                                <div className="bg-indigo-600 mt-2 px-3 py-1 rounded-xl text-white font-bold mx-auto w-20 border-2 border-white">PLAY</div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* AR Scanner & How It Works */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-purple-600/40 backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 border-purple-300">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-xl font-bold text-white flex items-center">
                                <Target className="text-purple-300 mr-2 animate-pulse" size={28} />
                                AR SCANNER
                            </h3>
                            <span className="text-base bg-purple-800 text-white px-3 py-1 rounded-xl font-bold border-2 border-white">READY!</span>
                        </div>

                        <div className="bg-indigo-900/50 h-64 rounded-xl flex items-center justify-center border-4 border-indigo-400">
                            <div className="text-center">
                                <div className="w-24 h-24 mx-auto bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center mb-4 border-4 border-white animate-pulse">
                                    <Glasses size={48} className="text-white" />
                                </div>
                                <p className="text-white font-bold text-lg mb-2">Point camera at any flat surface!</p>
                                <button className="bg-gradient-to-r from-pink-500 to-purple-500 text-white px-6 py-2 rounded-xl font-bold border-2 border-white hover:from-purple-500 hover:to-pink-500 transition-all">
                                    START MAGIC CAMERA
                                </button>
                            </div>
                        </div>
                    </div>

                    <div className="bg-green-600/40 backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 border-green-300">
                        <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center text-white">
                                <div className="bg-green-500 p-3 rounded-xl mr-3 border-2 border-white">
                                    <Rocket size={24} className="text-white" />
                                </div>
                                <h3 className="font-bold text-xl">HOW AR WORKS</h3>
                            </div>
                            <div className="bg-yellow-600 text-white px-3 py-2 rounded-xl text-base font-bold border-2 border-white">COOL TECH!</div>
                        </div>

                        <div className="space-y-3">
                            <div className="bg-gradient-to-r from-green-500/40 to-green-400/20 p-4 rounded-xl border-4 border-green-400 flex items-center">
                                <div className="w-12 h-12 bg-green-600 rounded-full flex items-center justify-center mr-3 border-2 border-white">
                                    <span className="text-white text-2xl font-bold">1</span>
                                </div>
                                <div>
                                    <p className="text-white font-bold text-lg">Point your device at a flat surface</p>
                                </div>
                            </div>

                            <div className="bg-gradient-to-r from-blue-500/40 to-blue-400/20 p-4 rounded-xl border-4 border-blue-400 flex items-center">
                                <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center mr-3 border-2 border-white">
                                    <span className="text-white text-2xl font-bold">2</span>
                                </div>
                                <div>
                                    <p className="text-white font-bold text-lg">Tap to place the 3D object</p>
                                </div>
                            </div>

                            <div className="bg-gradient-to-r from-purple-500/40 to-purple-400/20 p-4 rounded-xl border-4 border-purple-400 flex items-center">
                                <div className="w-12 h-12 bg-purple-600 rounded-full flex items-center justify-center mr-3 border-2 border-white">
                                    <span className="text-white text-2xl font-bold">3</span>
                                </div>
                                <div>
                                    <p className="text-white font-bold text-lg">Interact and learn with 3D magic!</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Special AR Feature */}
                <div className="mt-4 bg-gradient-to-br from-yellow-600/60 to-orange-600/40 backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 border-yellow-400 transform hover:scale-105 transition-all animate-pulse">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center">
                            <div className="bg-yellow-500 p-3 rounded-xl mr-3 border-2 border-white">
                                <Star size={28} className="text-white" />
                            </div>
                            <div>
                                <h3 className="text-xl font-bold text-white">SPECIAL AR CHALLENGE!</h3>
                                <p className="text-white font-bold">Find all 5 hidden objects in the Solar System!</p>
                            </div>
                        </div>
                        <div className="bg-orange-600 text-white px-4 py-2 rounded-xl text-lg font-bold border-2 border-white cursor-pointer hover:bg-orange-500">PLAY NOW!</div>
                    </div>
                </div>

                {/* Recently Unlocked Achievements */}
                <div className="mt-4 bg-pink-600/40 backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 border-pink-300">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-xl font-bold text-white flex items-center">
                            <BookOpen className="text-pink-300 mr-2" size={28} />
                            ACHIEVEMENTS UNLOCKED!
                        </h3>
                        <div className="bg-pink-800 text-white px-3 py-1 rounded-xl flex items-center font-bold border-2 border-white">
                            <Star className="text-yellow-300 mr-1" size={20} />
                            <span className="text-base">3/10</span>
                        </div>
                    </div>
                    <div className="grid grid-cols-3 gap-3">
                        <div className="bg-gradient-to-br from-pink-600/40 to-purple-600/20 p-3 rounded-xl text-center border-4 border-pink-400">
                            <div className="bg-gradient-to-r from-pink-500 to-purple-500 w-16 h-16 mx-auto rounded-full flex items-center justify-center mb-2 border-4 border-white">
                                <span className="text-3xl">🪐</span>
                            </div>
                            <h4 className="text-white font-bold text-lg">Space Explorer</h4>
                        </div>
                        <div className="bg-gradient-to-br from-blue-600/40 to-cyan-600/20 p-3 rounded-xl text-center border-4 border-blue-400">
                            <div className="bg-gradient-to-r from-blue-500 to-cyan-500 w-16 h-16 mx-auto rounded-full flex items-center justify-center mb-2 border-4 border-white">
                                <span className="text-3xl">🦕</span>
                            </div>
                            <h4 className="text-white font-bold text-lg">Dino Friend</h4>
                        </div>
                        <div className="bg-gradient-to-br from-green-600/40 to-teal-600/20 p-3 rounded-xl text-center border-4 border-green-400">
                            <div className="bg-gradient-to-r from-green-500 to-teal-500 w-16 h-16 mx-auto rounded-full flex items-center justify-center mb-2 border-4 border-white">
                                <span className="text-3xl">🧩</span>
                            </div>
                            <h4 className="text-white font-bold text-lg">Puzzle Master</h4>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default BalVidyaAugmentedRealityLearning;