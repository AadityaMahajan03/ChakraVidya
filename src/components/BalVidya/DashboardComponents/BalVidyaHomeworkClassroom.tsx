import React, { useState, useEffect } from 'react';
import { BookMarked, Users, Rocket, Star, Zap, Calendar, Moon, Sun, Award, Gift } from "lucide-react";

// Constants for all the data
const MISSION_DATA = [
    {
        id: 1,
        title: "Math Galaxy Quest",
        description: "Multiplication tables mastery",
        icon: "Rocket",
        status: "Mission Complete! ✨",
        progress: "100/100 points earned",
        colorClass: "green"
    },
    {
        id: 2,
        title: "Science Space Project",
        description: "Solar system model creation",
        icon: "Star",
        status: "Launch Tomorrow 🌠",
        progress: "75/100 points achieved",
        colorClass: "yellow"
    },
    {
        id: 3,
        title: "English Cosmic Essay",
        description: "My favorite space adventure",
        icon: "Zap",
        status: "Needs Attention! ⚠️",
        progress: "Due 2 days ago",
        colorClass: "red"
    },
    {
        id: 4,
        title: "History Time Travel",
        description: "Ancient civilizations research",
        icon: "Moon",
        status: "New Mission! 🔍",
        progress: "Due in 5 days",
        colorClass: "purple"
    }
];

const PROGRESS_DATA = [
    { subject: "Math Galaxy", percentage: 85 },
    { subject: "Science Expedition", percentage: 70 },
    { subject: "English Nebula", percentage: 45 },
    { subject: "History Wormhole", percentage: 20 }
];

const ACADEMY_DATA = [
    {
        id: 1,
        title: "Science Explorer Class",
        description: "Learn about planets and gravity experiments",
        icon: "Rocket",
        points: 50,
        time: "10:00 AM - Today",
        status: "LIVE NOW",
        statusColorClass: "cyan",
        action: "JOIN CLASS",
        colorClass: "blue"
    },
    {
        id: 2,
        title: "Math Adventure Class",
        description: "Solve exciting number puzzles and challenges",
        icon: "Star",
        points: 75,
        time: "11:30 AM - Today",
        status: "UPCOMING",
        statusColorClass: "purple",
        action: "SET REMINDER",
        colorClass: "purple"
    },
    {
        id: 3,
        title: "English Galaxy Class",
        description: "Write creative stories about space adventures",
        icon: "BookMarked",
        points: 60,
        time: "9:00 AM - Tomorrow",
        status: "TOMORROW",
        statusColorClass: "pink",
        action: "PREVIEW MATERIAL",
        colorClass: "pink"
    }
];

const REWARDS_DATA = [
    {
        title: "Space Explorer Badge",
        description: "Complete 5 more missions!",
        icon: "Sun",
        colorClass: "amber"
    },
    {
        title: "Rocket Pilot License",
        description: "Earn 200 more points!",
        icon: "Rocket",
        colorClass: "green"
    },
    {
        title: "Moon Voyager",
        description: "Attend 3 more classes!",
        icon: "Moon",
        colorClass: "blue"
    },
    {
        title: "Comet Catcher",
        description: "Submit all assignments!",
        icon: "Star",
        colorClass: "purple"
    }
];

// Icon mapping component for dynamic icon rendering
const IconComponent = ({ name, className, size }) => {
    const icons = {
        Rocket: Rocket,
        Star: Star,
        Zap: Zap,
        Moon: Moon,
        BookMarked: BookMarked,
        Sun: Sun
    };

    const IconToRender = icons[name];
    return IconToRender ? <IconToRender className={className} size={size} /> : null;
};

const BalVidyaHomeworkClassroom = () => {
    const [animationFrame, setAnimationFrame] = useState(0);
    const [starPositions, setStarPositions] = useState([]);

    // Generate random star positions
    useEffect(() => {
        const stars = [];
        for (let i = 0; i < 50; i++) {
            stars.push({
                top: Math.random() * 100 + '%',
                left: Math.random() * 100 + '%',
                size: Math.random() * 3 + 1 + 'px',
                animationDelay: Math.random() * 5 + 's',
                animationDuration: (Math.random() * 5 + 3) + 's'
            });
        }
        setStarPositions(stars);
    }, []);

    // Animation frame update for floating elements
    useEffect(() => {
        const animationInterval = setInterval(() => {
            setAnimationFrame(prev => prev + 1);
        }, 100);

        return () => clearInterval(animationInterval);
    }, []);

    // Helper function to get color classes
    const getColorClass = (type, colorClass) => {
        const colorMap = {
            bg: {
                green: "from-green-500/40 to-green-400/20",
                yellow: "from-yellow-500/40 to-yellow-400/20",
                red: "from-red-500/40 to-red-400/20",
                purple: "from-purple-500/40 to-purple-400/20",
                blue: "from-blue-500/40 to-blue-400/20",
                pink: "from-pink-500/40 to-pink-400/20",
                amber: "from-amber-500/40 to-amber-400/20",
                cyan: "from-cyan-500/40 to-cyan-400/20"
            },
            border: {
                green: "border-green-400",
                yellow: "border-yellow-400",
                red: "border-red-400",
                purple: "border-purple-400",
                blue: "border-blue-400",
                pink: "border-pink-400",
                amber: "border-amber-400",
                cyan: "border-cyan-400"
            },
            icon: {
                green: "bg-green-600",
                yellow: "bg-yellow-600",
                red: "bg-red-600",
                purple: "bg-purple-600",
                blue: "bg-blue-600",
                pink: "bg-pink-600",
                amber: "bg-amber-600",
                cyan: "bg-cyan-600"
            },
            button: {
                green: "bg-green-600 hover:bg-green-500",
                yellow: "bg-yellow-600 hover:bg-yellow-500",
                red: "bg-red-600 hover:bg-red-500",
                purple: "bg-purple-600 hover:bg-purple-500",
                blue: "bg-blue-600 hover:bg-blue-500",
                pink: "bg-pink-600 hover:bg-pink-500",
                amber: "bg-amber-600 hover:bg-amber-500",
                cyan: "bg-cyan-600 hover:bg-cyan-500"
            },
            statusBadge: {
                green: "bg-green-600",
                yellow: "bg-yellow-600",
                red: "bg-red-600",
                purple: "bg-purple-600",
                blue: "bg-blue-600",
                pink: "bg-pink-600",
                amber: "bg-amber-600",
                cyan: "bg-cyan-600"
            }
        };

        return colorMap[type][colorClass] || "";
    };

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
                                <Zap className="mr-3 text-yellow-300 animate-bounce" size={32} />
                                COSMIC MISSION CONTROL
                            </h1>
                            <p className="text-blue-200 text-xl">Track your space missions and attend galactic academy sessions! 🚀</p>
                        </div>
                        <div className="bg-indigo-700/80 p-3 rounded-xl flex items-center border-2 border-blue-300">
                            <div className="mr-3">
                                <Award className="text-yellow-300" size={28} />
                            </div>
                            <div>
                                <p className="text-base text-blue-200 font-bold">COSMIC EXPLORER</p>
                                <div className="w-36 h-4 bg-indigo-900 rounded-full mt-1 border border-blue-400">
                                    <div className="w-3/4 h-full bg-gradient-to-r from-yellow-300 to-orange-400 rounded-full animate-pulse"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Missions Section */}
                <div className="bg-blue-600/40 backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 border-blue-300 mb-6">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center text-white">
                            <div className="bg-blue-500 p-3 rounded-xl mr-3 border-2 border-white">
                                <Rocket size={24} className="text-white" />
                            </div>
                            <h3 className="font-bold text-xl">SPACE MISSIONS</h3>
                        </div>
                        <div className="bg-green-500 text-white px-3 py-2 rounded-xl text-base font-bold border-2 border-white">NEW MISSIONS!</div>
                    </div>

                    <div className="space-y-3">
                        {MISSION_DATA.map((mission) => (
                            <div key={mission.id} className={`bg-gradient-to-r ${getColorClass('bg', mission.colorClass)} p-4 rounded-xl border-4 ${getColorClass('border', mission.colorClass)} hover:scale-105 transition-all flex justify-between items-center`}>
                                <div className="flex items-center">
                                    <div className={`w-12 h-12 ${getColorClass('icon', mission.colorClass)} rounded-full flex items-center justify-center mr-3 border-2 border-white`}>
                                        <IconComponent name={mission.icon} className="text-white" size={24} />
                                    </div>
                                    <div>
                                        <h3 className="text-white font-bold text-lg">{mission.title}</h3>
                                        <p className="text-white/80 text-sm">{mission.description}</p>
                                    </div>
                                </div>
                                <div className="flex flex-col items-end">
                                    <span className={`${getColorClass('statusBadge', mission.colorClass)} text-white px-3 py-1 rounded-xl text-sm font-bold mb-1 border-2 border-white`}>
                                        {mission.status}
                                    </span>
                                    <span className="text-white/70 text-xs">{mission.progress}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Two-column layout */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Progress Tracker */}
                    <div className="bg-purple-600/40 backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 border-purple-300">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-xl font-bold text-white flex items-center">
                                <Award className="text-purple-300 mr-2" size={28} />
                                MISSION PROGRESS
                            </h3>
                            <span className="text-base bg-purple-800 text-white px-3 py-1 rounded-xl font-bold border-2 border-white">LEVEL 7</span>
                        </div>

                        <div className="space-y-4">
                            {PROGRESS_DATA.map((item, index) => (
                                <div key={index} className="bg-indigo-900/50 rounded-xl p-3 border-4 border-indigo-400">
                                    <div className="flex justify-between mb-1">
                                        <h3 className="text-white font-bold">{item.subject}</h3>
                                        <span className="text-white">{item.percentage}%</span>
                                    </div>
                                    <div className="w-full bg-indigo-900 rounded-full h-4 border border-indigo-300">
                                        <div
                                            className="bg-gradient-to-r from-cyan-500 to-blue-500 h-4 rounded-full"
                                            style={{ width: `${item.percentage}%` }}
                                        ></div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Cosmic Rewards */}
                    <div className="bg-green-600/40 backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 border-green-300">
                        <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center text-white">
                                <div className="bg-green-500 p-3 rounded-xl mr-3 border-2 border-white">
                                    <Gift size={24} className="text-white" />
                                </div>
                                <h3 className="font-bold text-xl">COSMIC REWARDS</h3>
                            </div>
                            <div className="bg-yellow-600 text-white px-3 py-2 rounded-xl text-base font-bold border-2 border-white">COLLECT ALL!</div>
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                            {REWARDS_DATA.map((reward, index) => (
                                <div key={index} className={`bg-gradient-to-r ${getColorClass('bg', reward.colorClass)} p-3 rounded-xl text-center border-4 ${getColorClass('border', reward.colorClass)}`}>
                                    <div className={`${getColorClass('icon', reward.colorClass)} w-16 h-16 mx-auto rounded-full flex items-center justify-center mb-2 border-4 border-white`}>
                                        <IconComponent name={reward.icon} className="text-white" size={28} />
                                    </div>
                                    <h4 className="text-white font-bold text-lg">{reward.title}</h4>
                                    <p className="text-white/80 text-sm">{reward.description}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Academy Classes */}
                <div className="mt-4 bg-pink-600/40 backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 border-pink-300">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-xl font-bold text-white flex items-center">
                            <Users className="text-pink-300 mr-2" size={28} />
                            SPACE ACADEMY
                        </h3>
                        <div className="bg-pink-800 text-white px-3 py-1 rounded-xl flex items-center font-bold border-2 border-white">
                            <Calendar className="text-yellow-300 mr-1" size={20} />
                            <span className="text-base">TODAY'S CLASSES</span>
                        </div>
                    </div>
                    <div className="space-y-4">
                        {ACADEMY_DATA.map((classItem) => (
                            <div key={classItem.id} className={`bg-gradient-to-r ${getColorClass('bg', classItem.colorClass)} p-4 rounded-xl border-4 ${getColorClass('border', classItem.colorClass)} hover:scale-102 transition-all`}>
                                <div className="flex justify-between items-center">
                                    <div className="flex items-center">
                                        <div className={`${getColorClass('icon', classItem.colorClass)} w-12 h-12 rounded-full flex items-center justify-center mr-3 border-2 border-white`}>
                                            <IconComponent name={classItem.icon} className="text-white" size={24} />
                                        </div>
                                        <div>
                                            <h3 className="text-white font-bold text-lg">{classItem.title}</h3>
                                            <p className="text-white/80 text-sm">{classItem.description}</p>
                                            <div className="flex items-center mt-1 text-white/80">
                                                <Calendar className="mr-1" size={14} />
                                                <span className="text-xs">{classItem.time}</span>
                                                <span className={`ml-2 px-2 py-0.5 bg-${classItem.statusColorClass}-600 rounded-lg text-white text-xs`}>
                                                    {classItem.status}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex flex-col items-end">
                                        <div className="bg-yellow-600 text-white px-2 py-1 rounded-lg text-xs font-bold mb-2 flex items-center border border-white">
                                            <Star className="mr-1" size={12} />
                                            {classItem.points} points
                                        </div>
                                        <button className={`${getColorClass('button', classItem.colorClass)} text-white px-3 py-1 rounded-lg text-sm font-bold border-2 border-white`}>
                                            {classItem.action}
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Action Button */}
                <div className="mt-6 flex justify-center">
                    <button className="relative group">
                        <div className="absolute -inset-0.5 bg-gradient-to-r from-pink-600 to-purple-600 rounded-xl blur opacity-75 group-hover:opacity-100 transition duration-300"></div>
                        <div className="relative px-6 py-3 bg-indigo-800 rounded-xl flex items-center gap-3 text-white font-bold text-xl border-2 border-white">
                            <Rocket size={24} />
                            BLAST OFF TO MISSION CONTROL! 🚀
                        </div>
                    </button>
                </div>

                {/* Animated Elements */}
                <div className="fixed top-1/4 left-1/4 w-8 h-1 bg-gradient-to-r from-transparent via-white to-blue-400 blur-sm animate-[shooting-star_6s_linear_infinite]" style={{ transform: 'rotate(-15deg)' }}></div>
                <div className="fixed top-3/4 right-1/4 w-6 h-1 bg-gradient-to-r from-transparent via-white to-purple-400 blur-sm animate-[shooting-star_8s_linear_infinite]" style={{ transform: 'rotate(15deg)', animationDelay: '2s' }}></div>
            </div>
        </div>
    );
};

export default BalVidyaHomeworkClassroom;