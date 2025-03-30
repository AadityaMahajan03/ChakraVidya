import React, { useState, useEffect } from 'react';
import { Play, Calendar, Star, Music, Palette, BookOpen, Rocket, Zap, Gift } from "lucide-react";

const BalVidyaFunActivities = () => {
    const [animationFrame, setAnimationFrame] = useState(0);

    // Animation frame update for floating elements
    useEffect(() => {
        const animationInterval = setInterval(() => {
            setAnimationFrame(prev => prev + 1);
        }, 100);

        return () => clearInterval(animationInterval);
    }, []);

    const quizCategories = [
        {
            title: "🧠 Space Brain Teasers",
            description: "Test your knowledge about planets, stars and space with fun cosmic questions!",
            icon: <Rocket className="text-white" size={24} />,
            level: "Level 1-3",
            points: "40 Stars",
            color: "from-blue-600 to-cyan-500"
        },
        {
            title: "🦖 Dino Discovery Quiz",
            description: "Travel back in time and learn about amazing dinosaurs! Collect dino badges!",
            icon: <BookOpen className="text-white" size={24} />,
            level: "Level 1-4",
            points: "50 Stars",
            color: "from-green-500 to-teal-500"
        },
        {
            title: "🌈 Science Wizard",
            description: "Amazing experiments and cool science facts! Learn how things work in a fun way!",
            icon: <Zap className="text-white" size={24} />,
            level: "Level 2-5",
            points: "60 Stars",
            color: "from-purple-600 to-fuchsia-600"
        },
        {
            title: "🎨 Art Challenge",
            description: "Show off your creativity with fun drawing challenges and colorful quizzes!",
            icon: <Palette className="text-white" size={24} />,
            level: "Level 1-3",
            points: "45 Stars",
            color: "from-red-600 to-pink-600"
        },
        {
            title: "🎵 Music Magic",
            description: "Listen, guess and learn about different instruments and cool music sounds!",
            icon: <Music className="text-white" size={24} />,
            level: "Level 2-4",
            points: "55 Stars",
            color: "from-yellow-500 to-amber-600"
        },
        {
            title: "🧩 Super Puzzles",
            description: "Solve awesome puzzles that make your brain stronger! Unlock new puzzle packs!",
            icon: <Star className="text-white" size={24} />,
            level: "Level 1-5",
            points: "65 Stars",
            color: "from-indigo-600 to-violet-600"
        }
    ];

    return (
        <div className="min-h-screen w-full relative overflow-hidden bg-gradient-to-br from-indigo-600 via-purple-600 to-fuchsia-600 py-12 px-4">
            {/* Interactive background elements */}
            <div className="absolute inset-0 overflow-hidden">
                {/* Animated galaxy effect */}
                <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-purple-900 via-indigo-900 to-fuchsia-900 opacity-50"></div>

                {/* Floating planets */}
                <div className="absolute top-1/4 left-10 w-20 h-20 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 opacity-70 blur-sm"
                    style={{ transform: `translate(${Math.sin(animationFrame / 10) * 10}px, ${Math.cos(animationFrame / 10) * 10}px)` }}></div>
                <div className="absolute bottom-1/3 right-10 w-32 h-32 rounded-full bg-gradient-to-br from-red-500 to-orange-500 opacity-70 blur-sm"
                    style={{ transform: `translate(${Math.cos(animationFrame / 15) * 15}px, ${Math.sin(animationFrame / 15) * 15}px)` }}></div>
                <div className="absolute top-2/3 left-1/4 w-24 h-24 rounded-full bg-gradient-to-br from-emerald-500 to-green-500 opacity-70 blur-sm"
                    style={{ transform: `translate(${Math.sin(animationFrame / 12) * 12}px, ${Math.cos(animationFrame / 12) * 12}px)` }}></div>
            </div>

            {/* Content Container */}
            <div className="max-w-6xl mx-auto relative z-10">
                <div className="flex flex-col md:flex-row items-center justify-between mb-8 bg-black/30 backdrop-blur-md p-6 rounded-2xl border border-white/20">
                    <div className="relative mb-4 md:mb-0">
                        <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-red-500 via-yellow-500 via-green-500 via-blue-500 to-purple-500">
                            COSMIC QUIZ ADVENTURE <Zap className="inline text-yellow-400" size={28} />
                        </h1>
                        <div className="absolute -bottom-2 left-0 w-full h-1 bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600"></div>
                        <p className="text-cyan-300 mt-4 font-medium text-xl">Pick a quiz category and start your space journey! 🚀</p>
                    </div>

                    <div className="bg-gradient-to-r from-yellow-400 to-amber-500 text-black px-6 py-3 rounded-full font-bold text-lg flex items-center shadow-lg shadow-yellow-500/20">
                        <Star className="mr-2" size={20} /> 850 STARS COLLECTED! 🏆
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {quizCategories.map((category, index) => (
                        <div
                            key={index}
                            className={`bg-gradient-to-br ${category.color} rounded-2xl p-6 hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2 relative overflow-hidden`}
                        >
                            {/* Shine effect */}
                            <div className="absolute -inset-px bg-gradient-to-r from-white/20 to-transparent rounded-xl"></div>

                            <div className="relative">
                                <div className="flex items-center mb-4">
                                    <div className="p-3 mr-4 bg-white/10 backdrop-blur-sm rounded-xl">
                                        {category.icon}
                                    </div>
                                    <h3 className="font-bold text-white text-xl">{category.title}</h3>
                                </div>

                                <p className="text-white text-lg mb-5">{category.description}</p>

                                <div className="flex justify-between items-center">
                                    <span className="bg-black/30 backdrop-blur-sm text-white px-4 py-2 rounded-full text-sm font-bold">
                                        {category.level}
                                    </span>
                                    <span className="bg-black/30 backdrop-blur-sm text-white px-4 py-2 rounded-full text-sm font-bold flex items-center">
                                        <Star className="text-yellow-400 mr-2" size={16} />
                                        {category.points}
                                    </span>
                                </div>

                                <button className="mt-5 w-full bg-white/15 hover:bg-white/25 backdrop-blur-sm text-white font-bold py-3 rounded-xl transition-colors flex items-center justify-center border-2 border-white/30 text-lg">
                                    <Play size={20} className="mr-2" /> START QUIZ!
                                </button>
                            </div>
                        </div>
                    ))}
                </div>

                <div className="mt-10 flex justify-center">
                    <button className="relative group">
                        <div className="absolute -inset-0.5 bg-gradient-to-r from-pink-600 to-purple-600 rounded-full blur opacity-75 group-hover:opacity-100 transition duration-300"></div>
                        <div className="relative px-8 py-4 bg-black rounded-full flex items-center gap-3 text-white font-bold text-xl">
                            <Gift size={24} />
                            UNLOCK MORE QUIZ ADVENTURES! 🔍
                        </div>
                    </button>
                </div>
            </div>

            {/* Floating animated elements */}
            <div className="absolute top-20 right-10 w-8 h-8 bg-yellow-400 rounded-full animate-ping opacity-80" style={{ animationDuration: '3s', animationDelay: '1s' }}></div>
            <div className="absolute bottom-40 left-10 w-6 h-6 bg-green-400 rounded-full animate-ping opacity-80" style={{ animationDuration: '4s' }}></div>
            <div className="absolute top-40 left-1/4 w-7 h-7 bg-purple-400 rounded-full animate-ping opacity-80" style={{ animationDuration: '5s', animationDelay: '2s' }}></div>
        </div>
    );
};

export default BalVidyaFunActivities;