import React, { useState, useEffect } from 'react';
import { BookOpen, Star, Sparkles, Zap, Award } from 'lucide-react';

type NavItem = {
    id: string;
    label: string;
    icon: JSX.Element;
    points?: number;
    unlocked?: boolean;
    level?: number;
    color?: string;
};

type SideNavProps = {
    navItems: NavItem[];
    activeSection: string;
    setActiveSection: (id: string) => void;
    userPoints?: number;
    userName?: string;
};

export default function BalVidyaDashSideNav({
    navItems = [
        {
            id: "dashboard",
            label: "Space Dashboard",
            icon: <BookOpen size={20} />,
            unlocked: true,
            points: 5
        },
        {
            id: "missions",
            label: "Cosmic Missions",
            icon: <Star size={20} />,
            unlocked: true,
            points: 10
        },
        {
            id: "games",
            label: "Learning Games",
            icon: <Sparkles size={20} />,
            unlocked: true,
            points: 15
        },
        {
            id: "advanced",
            label: "Advanced Quests",
            icon: <Zap size={20} />,
            unlocked: false,
            level: 6
        }
    ],
    activeSection = "dashboard",
    emailOfUser = "",
    setActiveSection = () => { },
    userPoints = 250,
    userName = "Space Explorer"
}: SideNavProps) {
    const [hoverItem, setHoverItem] = useState<string | null>(null);
    const [isAnimating, setIsAnimating] = useState(false);
    const [sparklePosition, setSparklePosition] = useState({ x: 0, y: 0 });

    // Create animated effect when changing sections
    useEffect(() => {
        setIsAnimating(true);
        const timer = setTimeout(() => setIsAnimating(false), 1000);
        return () => clearTimeout(timer);
    }, [activeSection]);

    // Random sparkle position generator
    const generateSparkles = () => {
        setSparklePosition({
            x: Math.random() * 100,
            y: Math.random() * 100
        });
    };

    useEffect(() => {
        const interval = setInterval(generateSparkles, 2000);
        return () => clearInterval(interval);
    }, []);

    const [userNameToPass, setUserName] = useState("");

    useEffect(() => {
        const fetchUserName = async () => {
            if (!emailOfUser) return; // Ensure email is available

            try {
                const response = await fetch("http://localhost:5000/get-username", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ email: emailOfUser }),
                });

                const data = await response.json();
                if (response.ok) {
                    setUserName(data.fullName); // Set username in state
                } else {
                    console.error("Error fetching username:", data.message);
                }
            } catch (error) {
                console.error("Error:", error);
            }
        };

        fetchUserName();
    }, [emailOfUser]); // Runs when emailOfUser changes



    return (
        <div className="flex justify-center items-center h-screen bg-gray-900">
            <div className="w-80 h-full max-h-screen overflow-auto bg-gradient-to-b from-purple-900 via-indigo-800 to-blue-900 text-white rounded-2xl shadow-2xl border-4 border-indigo-400 relative">
                {/* Animated background elements */}
                <div className="absolute inset-0 overflow-hidden">
                    {/* Star field background */}
                    {[...Array(30)].map((_, i) => (
                        <div
                            key={i}
                            className="absolute bg-white rounded-full opacity-70"
                            style={{
                                width: `${Math.random() * 3 + 1}px`,
                                height: `${Math.random() * 3 + 1}px`,
                                left: `${Math.random() * 100}%`,
                                top: `${Math.random() * 100}%`,
                                animation: `twinkle ${Math.random() * 5 + 3}s ease-in-out infinite`
                            }}
                        />
                    ))}

                    {/* Floating orbs */}
                    <div className="absolute top-20 right-10 w-32 h-32 bg-pink-500 rounded-full opacity-10 blur-xl animate-float"></div>
                    <div className="absolute bottom-40 left-0 w-24 h-24 bg-blue-400 rounded-full opacity-20 blur-xl animate-float-delayed"></div>
                    <div className="absolute top-1/2 right-0 w-20 h-20 bg-purple-400 rounded-full opacity-15 blur-xl animate-pulse"></div>
                </div>

                {/* Header with animated elements */}
                <div className="p-6 relative z-10">
                    <div className="mb-4 text-center relative">
                        <div className="absolute -top-2 -left-2 w-full h-full bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-500 opacity-75 blur rounded-lg transform -rotate-2"></div>
                        <div className="relative bg-indigo-900 bg-opacity-60 p-3 rounded-xl border-2 border-indigo-400 backdrop-blur">
                            <h1 className="text-3xl font-black mb-1 flex justify-center items-center">
                                <BookOpen className="mr-2 text-yellow-300" />
                                <span className="text-transparent bg-clip-text bg-gradient-to-r from-yellow-200 via-green-300 to-cyan-400 font-extrabold">BalVidya</span>
                            </h1>

                            <div className="text-sm font-medium text-indigo-200">Welcome back, {userNameToPass}!</div>

                            {/* Animated sparkle that follows cursor (simulated) */}
                            <div
                                className="absolute w-6 h-6 pointer-events-none z-20"
                                style={{
                                    left: `${sparklePosition.x}%`,
                                    top: `${sparklePosition.y}%`,
                                    transition: 'all 0.5s ease-out'
                                }}
                            >
                                <Sparkles className="text-yellow-300" />
                            </div>
                        </div>
                    </div>

                    {/* Avatar and stats */}
                    <div className="flex items-center gap-4 mb-6">
                        <div className="w-16 h-16 rounded-full bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center border-2 border-indigo-300 shadow-lg relative overflow-hidden">
                            <div className="absolute inset-0 bg-gradient-to-b from-transparent to-indigo-900 opacity-30"></div>
                            <span className="text-2xl">👾</span>
                        </div>

                        <div className="flex-1">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center space-x-2">
                                    <Star className="h-5 w-5 text-yellow-300" fill="#facc15" />
                                    <span className="text-lg font-bold">{userPoints}</span>
                                </div>

                                <div className="px-3 py-1 bg-gradient-to-r from-purple-600 to-pink-600 rounded-full text-sm font-bold shadow-lg">
                                    Level 4
                                </div>
                            </div>

                            {/* XP Progress bar */}
                            <div className="mt-2 bg-indigo-950 rounded-full h-3 overflow-hidden p-0.5">
                                <div
                                    className="h-full bg-gradient-to-r from-green-400 via-cyan-500 to-blue-500 rounded-full relative"
                                    style={{ width: '65%' }}
                                >
                                    <div className="absolute inset-0 bg-white opacity-30 animate-pulse-slow"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Navigation items with enhanced animations */}
                <nav className="px-4 pb-6 relative z-10">
                    <ul className="space-y-3">
                        {navItems.map((item) => {
                            // Determine if item should appear "locked"
                            const isLocked = item.unlocked === false;
                            const isActive = activeSection === item.id;
                            const isHovered = hoverItem === item.id;

                            // Generate unique colors for each nav item if not provided
                            const itemColor = item.color || `hsl(${navItems.findIndex(i => i.id === item.id) * 30 + 200}, 70%, 60%)`;
                            const itemColorDark = item.color ? `${item.color}90` : `hsl(${navItems.findIndex(i => i.id === item.id) * 30 + 200}, 70%, 30%)`;

                            return (
                                <li
                                    key={item.id}
                                    onMouseEnter={() => setHoverItem(item.id)}
                                    onMouseLeave={() => setHoverItem(null)}
                                    className="relative"
                                >
                                    {/* Animated selection indicator */}
                                    {isActive && (
                                        <div className="absolute -left-4 top-1/2 transform -translate-y-1/2 h-10 w-2 bg-gradient-to-b from-cyan-400 to-blue-600 rounded-full animate-pulse"></div>
                                    )}

                                    <button
                                        onClick={() => !isLocked && setActiveSection(item.id)}
                                        disabled={isLocked}
                                        className={`
                                          relative w-full text-left px-4 py-3 rounded-xl transition-all duration-300
                                          ${isLocked ? 'opacity-60 cursor-not-allowed filter grayscale' : 'cursor-pointer'}
                                          ${isActive ? "bg-indigo-800 bg-opacity-90 font-bold shadow-xl border-2 border-opacity-60" : "hover:bg-indigo-700 hover:bg-opacity-50 border-2 border-transparent"}
                                        `}
                                        style={{
                                            boxShadow: isActive ? `0 0 15px ${itemColorDark}` : 'none',
                                            transform: isActive ? 'scale(1.05)' : isHovered ? 'scale(1.02)' : 'scale(1)'
                                        }}
                                    >
                                        {/* Background effects when active or hovered */}
                                        {(isActive || isHovered) && !isLocked && (
                                            <div
                                                className="absolute inset-0 rounded-xl opacity-20"
                                                style={{
                                                    background: `linear-gradient(135deg, ${itemColor}80, transparent)`,
                                                    animation: isAnimating && isActive ? 'pulse 1s ease-in-out' : 'none'
                                                }}
                                            ></div>
                                        )}

                                        <div className="flex items-center relative z-10">
                                            {/* Icon container with effects */}
                                            <div
                                                className={`
                                                  mr-3 p-2.5 rounded-xl transition-all relative overflow-hidden
                                                  ${isActive ? "text-white" : "text-indigo-100"}
                                                `}
                                                style={{
                                                    background: isActive ? `linear-gradient(45deg, ${itemColor}, ${itemColorDark})` : 'rgba(30, 30, 70, 0.3)',
                                                    boxShadow: isActive ? `0 0 10px ${itemColorDark}` : 'none'
                                                }}
                                            >
                                                {/* Animated background for active icon */}
                                                {isActive && (
                                                    <div className="absolute inset-0 bg-white opacity-20 animate-pulse-slow"></div>
                                                )}

                                                {item.icon}

                                                {/* Show level requirement if locked */}
                                                {isLocked && item.level && (
                                                    <div className="absolute -bottom-1 -right-1 flex items-center justify-center w-5 h-5 bg-gray-800 text-xs rounded-full border border-gray-600">
                                                        {item.level}
                                                    </div>
                                                )}
                                            </div>

                                            <div className="flex-1">
                                                <div className="flex items-center justify-between">
                                                    <span className={isActive ? 'text-white' : ''}>{item.label}</span>

                                                    {isLocked ? (
                                                        <span className="text-xs bg-gray-800 px-2 py-0.5 rounded-full flex items-center ml-1">
                                                            🔒 <span className="ml-1">Lvl {item.level}</span>
                                                        </span>
                                                    ) : (
                                                        isActive && <Zap className="h-4 w-4 text-yellow-300 animate-pulse ml-1" />
                                                    )}
                                                </div>

                                                {/* Show points reward if available */}
                                                {item.points && !isLocked && (
                                                    <div className="text-xs mt-0.5 flex items-center">
                                                        <Star className="h-3 w-3 text-yellow-300 mr-1" fill="#facc15" />
                                                        +{item.points} cosmic points
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </button>

                                    {/* Animated selection trail */}
                                    {isActive && (
                                        <div className="absolute top-0 -right-4 h-full w-1">
                                            {[...Array(5)].map((_, i) => (
                                                <div
                                                    key={i}
                                                    className="absolute w-1 rounded-full bg-cyan-400"
                                                    style={{
                                                        height: `${6 + i * 3}px`,
                                                        top: `${20 + i * 15}%`,
                                                        opacity: 0.7 - i * 0.15,
                                                        animation: `float-vertical ${1 + i * 0.5}s ease-in-out infinite alternate`
                                                    }}
                                                />
                                            ))}
                                        </div>
                                    )}
                                </li>
                            );
                        })}
                    </ul>
                </nav>

                {/* Achievement badges */}
                <div className="px-4 pb-4 relative z-10">
                    <div className="bg-indigo-900 bg-opacity-40 rounded-xl p-3 border border-indigo-700">
                        <div className="flex items-center justify-between mb-2">
                            <span className="font-bold flex items-center text-sm">
                                <Award className="h-4 w-4 mr-1 text-yellow-300" />
                                Recent Achievements
                            </span>
                            <span className="text-xs px-2 py-0.5 bg-indigo-700 rounded-full">3/12</span>
                        </div>

                        <div className="flex justify-between">
                            {[...Array(4)].map((_, i) => (
                                <div
                                    key={i}
                                    className={`w-14 h-14 rounded-full flex items-center justify-center relative 
                                      ${i < 3 ? 'bg-gradient-to-br from-yellow-400 to-amber-600' : 'bg-gray-800'}`}
                                >
                                    <div className={`absolute inset-1 rounded-full flex items-center justify-center bg-indigo-950 ${i < 3 ? '' : 'opacity-60'}`}>
                                        <span className="text-xl">{i < 3 ? ['🚀', '🔭', '🧪'][i] : '?'}</span>
                                    </div>
                                    {i < 3 && (
                                        <div className="absolute -bottom-1 -right-1 w-5 h-5 bg-green-500 rounded-full border-2 border-indigo-800 flex items-center justify-center">
                                            <Zap className="h-3 w-3 text-white" />
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Global style for animations */}
                <style jsx global>{`
                    @keyframes twinkle {
                      0%, 100% { opacity: 0.2; }
                      50% { opacity: 0.7; }
                    }
                    
                    @keyframes float {
                      0% { transform: translateY(0) translateX(0); }
                      50% { transform: translateY(-10px) translateX(5px); }
                      100% { transform: translateY(0) translateX(0); }
                    }
                    
                    @keyframes float-delayed {
                      0% { transform: translateY(0) translateX(0); }
                      40% { transform: translateY(-15px) translateX(-7px); }
                      100% { transform: translateY(0) translateX(0); }
                    }
                    
                    @keyframes float-vertical {
                      0% { transform: translateY(0); }
                      100% { transform: translateY(5px); }
                    }
                    
                    @keyframes pulse-slow {
                      0%, 100% { opacity: 0.1; }
                      50% { opacity: 0.3; }
                    }
                    
                    .animate-float {
                      animation: float 8s ease-in-out infinite;
                    }
                    
                    .animate-float-delayed {
                      animation: float-delayed 10s ease-in-out infinite;
                    }
                    
                    .animate-pulse-slow {
                      animation: pulse-slow 3s ease-in-out infinite;
                    }
                `}</style>
            </div>
        </div>
    );
}