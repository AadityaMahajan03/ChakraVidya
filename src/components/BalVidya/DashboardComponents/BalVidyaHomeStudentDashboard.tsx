import React, { useEffect, useState } from 'react';
import { Calendar, BookMarked, Award, Star, BookOpen, Play, Rocket, Target, Trophy, Zap, Gift } from "lucide-react";

export default function BalVidyaHomeStudentDashboard({ emailOfUser }) {
    const [user, setUser] = useState(null);
    const [activities, setActivities] = useState([]);

    //get function that would get the object of user
    const userObject = async () => {
        try {
            const response = await fetch("http://localhost:5000/get-user-object", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ email: emailOfUser }),
            });
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            const data = await response.json();
            const userVal = data.user;
            setUser(userVal);
            console.log("User object:", userVal);
        } catch (error) {
            console.error("Failed to get user object:", error);
        }
    }

    // write a function that would call backend to mark the task as done
    const fingerCount = async () => {
        try {
            const response = await fetch("http://localhost:5000/balVidya/home/math", {
                method: "GET",
            });
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            const data = await response.json();
            console.log("Task marked as done:", data);
        } catch (error) {
            console.error("Failed to mark task as done:", error);
        }
    }
    // write a function that would call backend to mark the task as done
    const senseOrgan = async () => {
        try {
            const response = await fetch("http://localhost:5000/balVidya/home/sense-organ", {
                method: "GET",
            });
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            const data = await response.json();
            console.log("Task marked as done:", data);
        } catch (error) {
            console.error("Failed to mark task as done:", error);
        }
    }

    // write a function that would call backend to mark the task as done
    const manners = async () => {
        try {
            const response = await fetch("http://localhost:5000/balVidya/home/manners", {
                method: "GET",
            });
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            const data = await response.json();
            console.log("Task marked as done:", data);
        } catch (error) {
            console.error("Failed to mark task as done:", error);
        }
    }
    // write a function that would call backend to mark the task as done
    const interactiveReader = async () => {
        try {
            const response = await fetch("http://localhost:5000/balVidya/home/interactiveReader", {
                method: "GET",
            });
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            const data = await response.json();
            console.log("Task marked as done:", data);
        } catch (error) {
            console.error("Failed to mark task as done:", error);
        }
    }


    const activitiesForKindergarden = [
        { name: "1 2 3!", onClick: fingerCount },
        { name: "Body!", onClick: senseOrgan },
        { name: "Manners!", onClick: manners },// No function for this one
        { name: "ABC", onClick: interactiveReader } // No function for this one

    ];

    const activitiesForGrade1AndAbove = [
        { name: "Count!", onClick: fingerCount },
        { name: "Body!", onClick: senseOrgan },
        { name: "Manners!", onClick: manners }, // No function for this one
        { name: "ABC", onClick: interactiveReader } // No function for this one
    ];
    useEffect(() => {
        userObject();
    }, []);

    useEffect(() => {
        if (user && user.grade === "Kindergarden") {
            setActivities(activitiesForKindergarden);
        }
        else {
            setActivities(activitiesForGrade1AndAbove);
        }
    }, [user]);


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
                                <Rocket className="mr-3 text-yellow-300 animate-bounce" size={32} />
                                SPACE HEROES!
                            </h1>
                            <p className="text-blue-200 text-xl">Hi there, Super Star! 🌟</p>
                        </div>
                        <div className="bg-indigo-700/80 p-3 rounded-xl flex items-center border-2 border-blue-300">
                            <div className="mr-3">
                                <Zap className="text-yellow-300" size={28} />
                            </div>
                            <div>
                                <p className="text-base text-blue-200 font-bold">SUPER POWER</p>
                                <div className="w-36 h-4 bg-indigo-900 rounded-full mt-1 border border-blue-400">
                                    <div className="w-2/3 h-full bg-gradient-to-r from-yellow-300 to-orange-400 rounded-full animate-pulse"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Fun Progress Trackers */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div className="bg-blue-600/40 backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 border-blue-300 transform hover:scale-105 transition-all">
                        <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center text-white">
                                <div className="bg-blue-500 p-3 rounded-xl mr-3 border-2 border-white">
                                    <Calendar size={24} className="text-white" />
                                </div>
                                <h3 className="font-bold text-xl">TODAY'S QUESTS</h3>
                            </div>
                            <div className="bg-green-500 text-white px-3 py-2 rounded-xl text-base font-bold border-2 border-white">3/5 DONE!</div>
                        </div>
                        <div className="space-y-2">
                            {activities.map((activity, index) => (
                                <div
                                    key={index}
                                    onClick={activity.onClick || undefined} // Prevent click event if no function
                                    className="bg-white/30 p-3 rounded-xl flex items-center border-2 border-green-300 cursor-pointer"
                                >
                                    <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center mr-3 border-2 border-white">
                                        <span className="text-white text-base font-bold">✓</span>
                                    </div>
                                    <span className="text-white text-lg font-bold">{activity.name}</span>
                                </div>
                            ))}
                        </div>

                    </div>

                    <div className="bg-purple-600/40 backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 border-purple-300 transform hover:scale-105 transition-all">
                        <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center text-white">
                                <div className="bg-purple-500 p-3 rounded-xl mr-3 border-2 border-white">
                                    <Target size={24} className="text-white" />
                                </div>
                                <h3 className="font-bold text-xl">COMING UP!</h3>
                            </div>
                            <div className="bg-purple-800 text-white px-3 py-2 rounded-xl text-base font-bold border-2 border-white animate-pulse">NEW!</div>
                        </div>
                        <div className="space-y-2">
                            <div className="bg-purple-500/40 p-3 rounded-xl flex items-center justify-between border-2 border-pink-300">
                                <div className="flex items-center">
                                    <Zap className="mr-2 text-yellow-300" size={24} />
                                    <span className="text-white text-lg font-bold">Number Blast!</span>
                                </div>
                                <span className="text-base bg-purple-800 text-white px-3 py-1 rounded-xl font-bold border-2 border-white">TODAY</span>
                            </div>
                            <div className="bg-white/30 p-3 rounded-xl flex items-center justify-between border-2 border-blue-300">
                                <div className="flex items-center">
                                    <BookOpen className="mr-2 text-blue-300" size={24} />
                                    <span className="text-white text-lg font-bold">Space Stories!</span>
                                </div>
                                <span className="text-base bg-indigo-600 text-white px-3 py-1 rounded-xl font-bold border-2 border-white">SOON</span>
                            </div>
                        </div>
                    </div>

                    <div className="bg-yellow-500/40 backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 border-yellow-300 transform hover:scale-105 transition-all">
                        <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center text-white">
                                <div className="bg-yellow-500 p-3 rounded-xl mr-3 border-2 border-white">
                                    <Trophy size={24} className="text-white" />
                                </div>
                                <h3 className="font-bold text-xl">COOL STUFF!</h3>
                            </div>
                            <div className="bg-yellow-600 text-white px-3 py-2 rounded-xl text-base font-bold border-2 border-white">LEVEL 7</div>
                        </div>
                        <div className="text-center py-2">
                            <div className="inline-block bg-gradient-to-r from-yellow-300 to-orange-400 p-4 rounded-full mb-2 border-4 border-white animate-bounce">
                                <Star className="text-white" size={40} />
                            </div>
                            <h4 className="text-white font-bold text-2xl">5-Day SUPER Streak!</h4>
                            <p className="text-blue-100 text-base font-bold">Keep going to get prizes!</p>
                        </div>
                        <div className="mt-2 bg-indigo-900/50 rounded-xl p-2 border-2 border-blue-300">
                            <div className="flex justify-between text-base text-white font-bold">
                                <span>Next Prize:</span>
                                <span>Almost there!</span>
                            </div>
                            <div className="w-full h-6 bg-indigo-700 rounded-full mt-1 border-2 border-white p-1">
                                <div className="w-8/12 h-full bg-gradient-to-r from-green-300 to-green-500 rounded-full animate-pulse"></div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Fun Activities and Prizes */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-green-600/40 backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 border-green-300">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-xl font-bold text-white flex items-center">
                                <Play className="text-green-300 mr-2 animate-pulse" size={28} />
                                MY ADVENTURES!
                            </h3>
                            <span className="text-base bg-green-800 text-white px-3 py-1 rounded-xl font-bold border-2 border-white cursor-pointer hover:bg-green-700">SEE ALL</span>
                        </div>
                        <div className="space-y-3">
                            <div className="bg-gradient-to-r from-green-500/40 to-green-400/20 p-4 rounded-xl border-4 border-green-400 flex items-center">
                                <BookOpen className="text-green-300 mr-3" size={28} />
                                <div>
                                    <p className="text-white font-bold text-lg">ABC Book Finished!</p>
                                    <div className="flex items-center">
                                        <div className="bg-yellow-400 px-2 py-1 rounded-lg mr-2 text-base font-bold">+250</div>
                                        <p className="text-white font-bold">Today</p>
                                    </div>
                                </div>
                            </div>
                            <div className="bg-gradient-to-r from-blue-500/40 to-blue-400/20 p-4 rounded-xl border-4 border-blue-400 flex items-center">
                                <Play className="text-blue-300 mr-3" size={28} />
                                <div>
                                    <p className="text-white font-bold text-lg">Number Game Win!</p>
                                    <div className="flex items-center">
                                        <div className="bg-yellow-400 px-2 py-1 rounded-lg mr-2 text-base font-bold">+120</div>
                                        <p className="text-white font-bold">Yesterday</p>
                                    </div>
                                </div>
                            </div>
                            <div className="bg-gradient-to-r from-purple-500/40 to-purple-400/20 p-4 rounded-xl border-4 border-purple-400 flex items-center">
                                <Award className="text-purple-300 mr-3" size={28} />
                                <div>
                                    <p className="text-white font-bold text-lg">Science Explorer Badge!</p>
                                    <div className="flex items-center">
                                        <div className="bg-yellow-400 px-2 py-1 rounded-lg mr-2 text-base font-bold">+350</div>
                                        <p className="text-white font-bold">2 days ago</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="bg-pink-600/40 backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 border-pink-300">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-xl font-bold text-white flex items-center">
                                <Gift className="text-pink-300 mr-2 animate-bounce" size={28} />
                                PRIZE SHOP!
                            </h3>
                            <div className="bg-pink-800 text-white px-3 py-1 rounded-xl flex items-center font-bold border-2 border-white">
                                <Star className="text-yellow-300 mr-1" size={20} />
                                <span className="text-base">2500</span>
                            </div>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                            <div className="bg-gradient-to-br from-pink-600/40 to-purple-600/20 p-3 rounded-xl text-center border-4 border-pink-400 hover:border-yellow-300 cursor-pointer transition-all">
                                <div className="bg-gradient-to-r from-pink-500 to-purple-500 w-16 h-16 mx-auto rounded-full flex items-center justify-center mb-2 border-4 border-white animate-pulse">
                                    <Rocket className="text-white" size={28} />
                                </div>
                                <h4 className="text-white font-bold text-lg">Space Rocket!</h4>
                                <div className="bg-yellow-500 mx-auto w-20 px-2 py-1 rounded-xl text-white font-bold mt-1 border-2 border-white">1000</div>
                            </div>
                            <div className="bg-gradient-to-br from-blue-600/40 to-cyan-600/20 p-3 rounded-xl text-center border-4 border-blue-400 hover:border-yellow-300 cursor-pointer transition-all">
                                <div className="bg-gradient-to-r from-blue-500 to-cyan-500 w-16 h-16 mx-auto rounded-full flex items-center justify-center mb-2 border-4 border-white animate-pulse">
                                    <Trophy className="text-white" size={28} />
                                </div>
                                <h4 className="text-white font-bold text-lg">Gold Trophy!</h4>
                                <div className="bg-yellow-500 mx-auto w-20 px-2 py-1 rounded-xl text-white font-bold mt-1 border-2 border-white">2000</div>
                            </div>
                        </div>

                        {/* Add Pet Companion */}
                        <div className="mt-4 bg-gradient-to-br from-cyan-600/40 to-blue-600/20 p-3 rounded-xl border-4 border-cyan-400">
                            <div className="flex items-center">
                                <div className="w-16 h-16 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full border-4 border-white flex items-center justify-center mr-3">
                                    <span className="text-3xl">🐉</span>
                                </div>
                                <div>
                                    <h4 className="text-white font-bold text-lg">Your Space Dragon!</h4>
                                    <div className="flex items-center mt-1">
                                        <div className="w-32 h-4 bg-cyan-800 rounded-full border border-white">
                                            <div className="w-1/2 h-full bg-cyan-400 rounded-full"></div>
                                        </div>
                                        <span className="text-white ml-2 font-bold">Level 3</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Daily Challenge - Special Feature */}
                <div className="mt-4 bg-gradient-to-br from-yellow-600/60 to-orange-600/40 backdrop-blur-md p-4 rounded-3xl shadow-lg border-4 border-yellow-400 transform hover:scale-105 transition-all animate-pulse">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center">
                            <div className="bg-yellow-500 p-3 rounded-xl mr-3 border-2 border-white">
                                <Zap size={28} className="text-white" />
                            </div>
                            <div>
                                <h3 className="text-xl font-bold text-white">DAILY SUPER CHALLENGE!</h3>
                                <p className="text-white font-bold">Solve the puzzle to earn DOUBLE POINTS!</p>
                            </div>
                        </div>
                        <div className="bg-orange-600 text-white px-4 py-2 rounded-xl text-lg font-bold border-2 border-white cursor-pointer hover:bg-orange-500">PLAY NOW!</div>
                    </div>
                </div>
            </div>
        </div>
    );
}