import React, { useState, useEffect } from 'react';
import { BookOpen, Microscope, PaintBucket, Music, Globe, Calculator } from "lucide-react";

export default function BalVidyaLearningModules() {
    const [hoveredCard, setHoveredCard] = useState(null);
    const [animationFrame, setAnimationFrame] = useState(0);

    // Animation loop for background elements
    useEffect(() => {
        const timer = setInterval(() => {
            setAnimationFrame(prev => (prev + 1) % 100);
        }, 200);
        return () => clearInterval(timer);
    }, []);

    const subjects = [
        {
            name: "Math Fun",
            color: "bg-gradient-to-br from-blue-500 to-indigo-600",
            hoverColor: "from-blue-600 to-indigo-700",
            textColor: "text-white",
            icon: <Calculator className="h-8 w-8" />,
            emoji: "🔢",
            description: "Solve puzzles & win stars!",
            borderColor: "border-blue-300"
        },
        {
            name: "Science Wonders",
            color: "bg-gradient-to-br from-green-500 to-teal-600",
            hoverColor: "from-green-600 to-teal-700",
            textColor: "text-white",
            icon: <Microscope className="h-8 w-8" />,
            emoji: "🔬",
            description: "Discover amazing experiments!",
            borderColor: "border-green-300"
        },
        {
            name: "Story Time",
            color: "bg-gradient-to-br from-yellow-500 to-amber-600",
            hoverColor: "from-yellow-600 to-amber-700",
            textColor: "text-white",
            icon: <BookOpen className="h-8 w-8" />,
            emoji: "📚",
            description: "Magical adventures await!",
            borderColor: "border-yellow-300"
        },
        {
            name: "World Around Us",
            color: "bg-gradient-to-br from-purple-500 to-violet-600",
            hoverColor: "from-purple-600 to-violet-700",
            textColor: "text-white",
            icon: <Globe className="h-8 w-8" />,
            emoji: "🌎",
            description: "Explore exciting places!",
            borderColor: "border-purple-300"
        },
        {
            name: "Art & Craft",
            color: "bg-gradient-to-br from-pink-500 to-rose-600",
            hoverColor: "from-pink-600 to-rose-700",
            textColor: "text-white",
            icon: <PaintBucket className="h-8 w-8" />,
            emoji: "🎨",
            description: "Create awesome projects!",
            borderColor: "border-pink-300"
        },
        {
            name: "Music & Rhymes",
            color: "bg-gradient-to-br from-orange-500 to-red-600",
            hoverColor: "from-orange-600 to-red-700",
            textColor: "text-white",
            icon: <Music className="h-8 w-8" />,
            emoji: "🎵",
            description: "Sing, dance & have fun!",
            borderColor: "border-orange-300"
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

                {/* Magic sparkles */}
                {[...Array(30)].map((_, i) => (
                    <div
                        key={`sparkle-${i}`}
                        className="absolute text-white text-opacity-90"
                        style={{
                            top: `${Math.random() * 100}%`,
                            left: `${Math.random() * 100}%`,
                            fontSize: `${Math.random() * 16 + 8}px`,
                            opacity: 0.5 + Math.sin((animationFrame + i) / 10) * 0.5,
                            transform: `rotate(${animationFrame * 2 + i * 30}deg)`
                        }}
                    >
                        {['✨', '⭐', '💫', '✴️', '🌟'][i % 5]}
                    </div>
                ))}

                {/* Fun floating elements */}
                {['🚀', '🌈', '🎒', '🔍', '🧩', '🎯', '🎨', '🎵', '🎮', '🧠', '📝', '🎭'].map((emoji, i) => (
                    <div
                        key={`float-${i}`}
                        className="absolute text-3xl sm:text-4xl"
                        style={{
                            top: `${10 + (i % 8) * 10}%`,
                            left: `${5 + (i % 6) * 15}%`,
                            transform: `translateY(${Math.sin((animationFrame + i * 10) / 15) * 10}px) rotate(${Math.sin((animationFrame + i) / 20) * 20}deg)`,
                            opacity: 0.8 + Math.sin(animationFrame / 10 + i) * 0.2,
                            filter: `blur(${Math.abs(Math.sin(animationFrame / 30 + i) * 0.5)}px)`
                        }}
                    >
                        {emoji}
                    </div>
                ))}

                {/* Rainbow path */}
                <div className="absolute top-1/2 left-0 right-0 h-10 bg-gradient-to-r from-red-500 via-yellow-500 via-green-500 via-blue-500 to-purple-500 opacity-30 transform -rotate-3 blur-md"></div>
                <div className="absolute top-1/3 left-0 right-0 h-6 bg-gradient-to-r from-purple-500 via-pink-500 via-red-500 via-orange-500 to-yellow-500 opacity-20 transform rotate-2 blur-md"></div>

                {/* Animated clouds */}
                <div className="absolute top-20 w-full">
                    <div className="relative h-16">
                        {[...Array(4)].map((_, i) => (
                            <div
                                key={`cloud-${i}`}
                                className="absolute h-16 w-32 bg-white opacity-20 rounded-full blur-md"
                                style={{
                                    left: `${((animationFrame / 2 + i * 100) % 140) - 20}%`,
                                    top: `${i * 8}px`
                                }}
                            ></div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="max-w-5xl mx-auto relative">
                {/* Shooting star animation */}
                <div
                    className="absolute w-20 h-1 bg-white blur-sm transform rotate-45"
                    style={{
                        top: `${20 + Math.sin(animationFrame / 20) * 10}%`,
                        left: `${(animationFrame * 2) % 100}%`,
                        opacity: animationFrame % 50 < 25 ? 0.8 : 0
                    }}
                ></div>

                {/* Colorful title section */}
                <div className="relative my-8 py-4">
                    <div className="absolute inset-0 bg-gradient-to-r from-blue-500/30 via-purple-500/30 to-pink-500/30 rounded-2xl backdrop-blur-sm"></div>

                    <div className="relative text-center z-10">
                        <div className="inline-block relative">
                            <h1 className="text-5xl font-extrabold mb-1 text-transparent bg-clip-text bg-gradient-to-r from-yellow-300 via-pink-300 to-cyan-300 drop-shadow-lg">
                                BalVidya Adventure Zone!
                            </h1>
                            <div className="absolute -top-6 -right-12 bg-gradient-to-r from-yellow-400 to-amber-500 text-amber-900 text-sm font-bold px-4 py-1 rounded-full transform rotate-12 shadow-lg border-2 border-amber-300 animate-pulse">
                                NEW!
                            </div>
                        </div>

                        <p className="text-white text-xl font-medium mt-3 mb-4 drop-shadow-md">
                            Choose your learning adventure and collect achievement badges! 🏆
                        </p>

                        <div className="flex justify-center space-x-2 mb-4">
                            {[...Array(5)].map((_, i) => (
                                <div
                                    key={i}
                                    className="relative group"
                                >
                                    <div className="absolute inset-0 bg-yellow-300 rounded-full blur-md opacity-50 group-hover:opacity-80 transition-opacity"></div>
                                    <div className={`relative w-8 h-8 text-yellow-300 transform transition-transform duration-500 ${hoveredCard === i ? 'scale-125' : ''
                                        }`}>
                                        ⭐
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Main grid of subjects */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 relative z-10">
                    {subjects.map((subject, index) => (
                        <div
                            key={index}
                            className="group relative"
                        >
                            {/* Glow effect on hover */}
                            <div className={`absolute inset-0 rounded-2xl blur-md transition-opacity ${subject.color} opacity-0 group-hover:opacity-70`}></div>

                            <div
                                className={`${subject.color} 
                  ${hoveredCard === index ? `bg-gradient-to-br ${subject.hoverColor}` : ''} 
                  ${subject.textColor} p-6 rounded-2xl shadow-xl backdrop-blur-sm transform transition-all duration-300 
                  ${hoveredCard === index ? 'scale-105 -rotate-2' : 'group-hover:scale-102'} 
                  cursor-pointer relative overflow-hidden border-3 ${subject.borderColor} z-10`}
                                onMouseEnter={() => setHoveredCard(index)}
                                onMouseLeave={() => setHoveredCard(null)}
                            >
                                {/* Animated swirl in background */}
                                <div className="absolute inset-0 overflow-hidden opacity-20">
                                    {[...Array(3)].map((_, i) => (
                                        <div
                                            key={`swirl-${i}`}
                                            className="absolute w-full h-full rounded-full border-4 border-white border-opacity-40"
                                            style={{
                                                top: '50%',
                                                left: '50%',
                                                width: `${200 + i * 20}%`,
                                                height: `${200 + i * 20}%`,
                                                transform: `translate(-50%, -50%) rotate(${animationFrame * (i + 1)}deg)`,
                                                borderRadius: '40% 60% 60% 40% / 70% 30% 70% 30%'
                                            }}
                                        ></div>
                                    ))}
                                </div>

                                <div className="flex items-center justify-between mb-4 relative z-10">
                                    <div className="bg-white rounded-full p-3 shadow-xl border-2 border-white">
                                        {subject.icon}
                                    </div>
                                    <div className="relative">
                                        <div className="absolute inset-0 bg-white rounded-full opacity-30 blur-sm group-hover:opacity-50 transition-opacity"></div>
                                        <span className="relative text-3xl transform transition-transform duration-300 hover:scale-125 hover:rotate-12">
                                            {subject.emoji}
                                        </span>
                                    </div>
                                </div>

                                <h3 className="font-bold text-xl mb-2 drop-shadow-md">{subject.name}</h3>
                                <p className="text-sm opacity-90 font-medium">{subject.description}</p>

                                <div className="mt-4 flex justify-between items-center">
                                    <span className="bg-white bg-opacity-30 px-3 py-1 rounded-full text-xs font-bold border border-white border-opacity-50 backdrop-blur-sm">
                                        Level {index + 1}
                                    </span>
                                    <div className="flex space-x-1">
                                        {[...Array(3)].map((_, i) => (
                                            <div
                                                key={i}
                                                className={`w-3 h-3 rounded-full transition-colors duration-300 ${i <= index % 3 ? 'bg-white' : 'bg-white bg-opacity-40'
                                                    }`}
                                            ></div>
                                        ))}
                                    </div>
                                </div>

                                {/* "Play" corner with glow */}
                                <div className={`absolute bottom-0 right-0 bg-white bg-opacity-30 p-2 rounded-tl-xl transform translate-y-full translate-x-full transition-all duration-500 backdrop-blur-sm ${hoveredCard === index ? 'translate-y-0 translate-x-0' : ''
                                    }`}>
                                    <span className="text-sm font-bold">PLAY</span>
                                </div>

                                {/* Sparkling effect on hover */}
                                {hoveredCard === index && [...Array(5)].map((_, i) => (
                                    <div
                                        key={`sparkle-${i}`}
                                        className="absolute text-white text-lg"
                                        style={{
                                            top: `${Math.random() * 100}%`,
                                            left: `${Math.random() * 100}%`,
                                            opacity: Math.random() * 0.5 + 0.5,
                                            transform: `rotate(${Math.random() * 360}deg)`
                                        }}
                                    >
                                        ✨
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>

                {/* Animated call to action */}
                <div className="mt-10 flex justify-center">
                    <div className="relative group">
                        {/* Button glow effect */}
                        <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-full blur-md opacity-70 group-hover:opacity-100 transition-opacity transform group-hover:scale-110"></div>

                        <button className="relative bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 text-white px-8 py-4 rounded-full font-bold text-lg shadow-xl group-hover:shadow-2xl transform transition-all group-hover:scale-105 flex items-center space-x-3 border-2 border-white border-opacity-50 z-10">
                            <span>Start Your Adventure!</span>
                            <span className="text-2xl animate-bounce">🚀</span>
                        </button>
                    </div>
                </div>

                {/* Animated characters at bottom with speech bubbles */}
                <div className="absolute -bottom-2 left-4 text-5xl transform -rotate-12 animate-bounce" style={{ animationDuration: '3s' }}>
                    <div className="relative">
                        <div className="absolute -top-10 -right-8 bg-white rounded-xl p-2 text-sm font-bold text-purple-800 transform -rotate-6">
                            Let's learn!
                        </div>
                        🧒
                    </div>
                </div>
                <div className="absolute -bottom-2 right-4 text-5xl transform rotate-12 animate-bounce" style={{ animationDuration: '2.5s' }}>
                    <div className="relative">
                        <div className="absolute -top-10 -left-8 bg-white rounded-xl p-2 text-sm font-bold text-purple-800 transform rotate-6">
                            So fun!
                        </div>
                        👧
                    </div>
                </div>
            </div>
        </div>
    );
}