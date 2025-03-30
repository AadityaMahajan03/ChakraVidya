import React, { useState } from "react";
import BalVidyaDashboard from "./BalVidyaDashboard";

function Login({ setActiveComponent, setEmailToPass }: {
    setActiveComponent: (comp: string) => void;
    setEmailToPass: (email: string) => void;
}) {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        let emailPass = email;
        try {
            const response = await fetch("http://localhost:5000/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, password }),
            });

            const data = await response.json();
            if (response.ok) {
                console.log("User Data:", data.user);
                setEmailToPass(emailPass); // Pass the email to dashboard component
                setActiveComponent("dashboard"); // Navigate to dashboard
            } else {
                alert(data.message);
            }
        } catch (error) {
            console.error("Error logging in:", error);
        }
    };



    return (
        <div className="min-h-screen py-6 flex flex-col justify-center bg-gradient-to-b from-blue-400 to-purple-500 relative overflow-hidden">
            {/* Decorative elements */}
            <div className="absolute top-10 left-10 text-6xl animate-bounce" style={{ animationDuration: '2s' }}>
                🪁
            </div>
            <div className="absolute bottom-10 right-10 text-6xl animate-pulse" style={{ animationDuration: '3s' }}>
                🎵
            </div>
            <div className="absolute top-1/4 right-10 text-5xl animate-spin" style={{ animationDuration: '10s' }}>
                ☸️
            </div>
            <div className="absolute bottom-1/4 left-10 text-5xl animate-pulse" style={{ animationDuration: '4s' }}>
                ✨
            </div>

            {/* Floating clouds */}
            <div className="absolute top-20 left-20 w-24 h-12 rounded-full bg-white opacity-60 animate-pulse"></div>
            <div className="absolute top-40 right-20 w-32 h-16 rounded-full bg-white opacity-70 animate-pulse" style={{ animationDuration: '5s' }}></div>

            <div className="container mx-auto px-4 relative z-10">
                <div className="max-w-md mx-auto bg-white rounded-2xl shadow-xl overflow-hidden">
                    <div className="px-6 py-8">
                        <div className="text-center mb-6">
                            <div className="flex justify-center">
                                <div className="text-5xl">🦚</div>
                                <div className="ml-3 text-5xl">🌺</div>
                            </div>
                            <h2 className="mt-4 text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">
                                Welcome, Little Star!
                            </h2>
                            <p className="mt-2 text-yellow-600">
                                Login to start your magical learning adventure!
                            </p>
                        </div>

                        <form onSubmit={handleLogin} className="space-y-6">
                            <div className="relative">
                                <label className="block text-purple-600 text-lg font-medium mb-2">
                                    Your Magic Email:
                                </label>
                                <div className="relative">
                                    <span className="absolute left-3 top-3 text-xl">✉️</span>
                                    <input
                                        type="email"
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        required
                                        className="w-full pl-10 pr-3 py-3 rounded-xl border-2 border-blue-300 focus:border-yellow-400 focus:ring focus:ring-yellow-200 transition-colors text-lg"
                                    />
                                </div>
                            </div>

                            <div className="relative">
                                <label className="block text-purple-600 text-lg font-medium mb-2">
                                    Secret Password:
                                </label>
                                <div className="relative">
                                    <span className="absolute left-3 top-3 text-xl">🔑</span>
                                    <input
                                        type="password"
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        required
                                        className="w-full pl-10 pr-3 py-3 rounded-xl border-2 border-blue-300 focus:border-yellow-400 focus:ring focus:ring-yellow-200 transition-colors text-lg"
                                    />
                                </div>
                            </div>

                            <button
                                type="submit"
                                className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-blue-500 to-purple-600 text-white text-lg font-bold shadow-lg hover:shadow-xl transform hover:-translate-y-1 transition-all duration-200"
                            >
                                Begin Adventure! 🚀
                            </button>
                        </form>

                        <div className="mt-6 text-center">
                            <button
                                onClick={() => setActiveComponent("register")}
                                className="text-lg text-blue-600 hover:text-purple-600 transition-colors duration-200 flex items-center justify-center mx-auto"
                            >
                                <span className="mr-2">🌟</span>
                                New friend? Join us!
                                <span className="ml-2">🌟</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

function Register({ setActiveComponent, setEmailToPass }: {
    setActiveComponent: (comp: string) => void;
    setEmailToPass: (email: string) => void;
}) {
    const [userType, setUserType] = useState<"Student" | "Teacher" | "Parent">("Student");
    const [grade, setGrade] = useState("");
    const [formData, setFormData] = useState({ fullName: "", email: "", password: "" });

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };


    const handleRegister = async (e: React.FormEvent) => {
        e.preventDefault();
        let emailPass = formData.email;
        try {
            const response = await fetch("http://localhost:5000/register", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    fullName: formData.fullName,
                    userType: userType,
                    grade: userType === "Student" ? grade : "", // Send grade only for students
                    email: formData.email,
                    password: formData.password,
                }),
            });

            const data = await response.json();

            if (response.ok) {
                alert("Registration successful! 🎉 Please log in.");
                setEmailToPass(emailPass); // Pass the email to login component
                setActiveComponent("login"); // Switch to login component
            } else {
                alert(`Error: ${data.error || "Something went wrong!"}`);
            }
        } catch (error) {
            alert("Failed to register. Please check your connection.");
        }
    };

    return (
        <div className="min-h-screen py-6 flex flex-col justify-center bg-gradient-to-b from-blue-400 to-purple-500 relative overflow-hidden">
            {/* Decorative elements */}
            <div className="absolute top-10 right-10 text-6xl animate-bounce" style={{ animationDuration: '2.5s' }}>
                🎭
            </div>
            <div className="absolute bottom-10 left-10 text-6xl animate-pulse" style={{ animationDuration: '2.8s' }}>
                🌈
            </div>
            <div className="absolute bottom-1/3 right-16 text-5xl animate-spin" style={{ animationDuration: '12s' }}>
                ☸️
            </div>
            <div className="absolute top-1/3 left-16 text-5xl animate-pulse" style={{ animationDuration: '4.5s' }}>
                🦋
            </div>

            {/* Floating clouds */}
            <div className="absolute top-32 right-36 w-24 h-12 rounded-full bg-white opacity-60 animate-pulse"></div>
            <div className="absolute bottom-36 left-20 w-32 h-16 rounded-full bg-white opacity-70 animate-pulse" style={{ animationDuration: '6s' }}></div>

            <div className="container mx-auto px-4 relative z-10">
                <div className="max-w-md mx-auto bg-white rounded-2xl shadow-xl overflow-hidden">
                    <div className="px-6 py-8">
                        <div className="text-center mb-6">
                            <div className="flex justify-center">
                                <div className="text-5xl">🏹</div>
                                <div className="ml-3 text-5xl">🪄</div>
                            </div>
                            <h2 className="mt-4 text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">
                                Join Our Learning Adventure!
                            </h2>
                            <p className="mt-2 text-yellow-600">
                                Tell us about yourself to begin your journey
                            </p>
                        </div>

                        <form onSubmit={handleRegister} className="space-y-5">
                            <div>
                                <label className="block text-purple-600 text-lg font-medium mb-2">
                                    Your Special Name:
                                </label>
                                <div className="relative">
                                    <span className="absolute left-3 top-3 text-xl">👤</span>
                                    <input
                                        type="text"
                                        name="fullName"
                                        value={formData.fullName}
                                        onChange={handleChange}
                                        required
                                        className="w-full pl-10 pr-3 py-3 rounded-xl border-2 border-blue-300 focus:border-yellow-400 focus:ring focus:ring-yellow-200 transition-colors text-lg"
                                    />
                                </div>
                            </div>

                            <div>
                                <label className="block text-purple-600 text-lg font-medium mb-2">
                                    I am a:
                                </label>
                                <div className="relative">
                                    <span className="absolute left-3 top-3 text-xl">
                                        {userType === "Student" ? "👨‍🎓" : userType === "Teacher" ? "👨‍🏫" : "👪"}
                                    </span>
                                    <select
                                        value={userType}
                                        onChange={(e) => setUserType(e.target.value as "Student" | "Teacher" | "Parent")}
                                        className="w-full pl-10 pr-3 py-3 rounded-xl border-2 border-blue-300 focus:border-yellow-400 focus:ring focus:ring-yellow-200 transition-colors text-lg appearance-none bg-white"
                                    >
                                        <option value="Student">Young Learner</option>
                                        <option value="Teacher">Wise Guide</option>
                                        <option value="Parent">Caring Parent</option>
                                    </select>
                                </div>
                            </div>

                            {userType === "Student" && (
                                <div>
                                    <label className="block text-purple-600 text-lg font-medium mb-2">
                                        My Learning Level:
                                    </label>
                                    <div className="relative">
                                        <span className="absolute left-3 top-3 text-xl">📚</span>
                                        <select
                                            name="grade"
                                            value={grade}
                                            onChange={(e) => setGrade(e.target.value)}
                                            required
                                            className="w-full pl-10 pr-3 py-3 rounded-xl border-2 border-blue-300 focus:border-yellow-400 focus:ring focus:ring-yellow-200 transition-colors text-lg appearance-none bg-white"
                                        >
                                            <option value="">Choose Your Level</option>
                                            <option value="Kindergarten">Little Sprouts (Kindergarten)</option>
                                            <option value="Grade 1">Curious Cubs (Grade 1)</option>
                                            <option value="Grade 2">Wonder Seekers (Grade 2)</option>
                                            <option value="Grade 3">Knowledge Explorers (Grade 3)</option>
                                            <option value="Grade 4">Wisdom Gatherers (Grade 4)</option>
                                            <option value="Grade 5">Young Sages (Grade 5)</option>
                                        </select>
                                    </div>
                                </div>
                            )}

                            <div>
                                <label className="block text-purple-600 text-lg font-medium mb-2">
                                    Magic Email Address:
                                </label>
                                <div className="relative">
                                    <span className="absolute left-3 top-3 text-xl">✉️</span>
                                    <input
                                        type="email"
                                        name="email"
                                        value={formData.email}
                                        onChange={handleChange}
                                        required
                                        className="w-full pl-10 pr-3 py-3 rounded-xl border-2 border-blue-300 focus:border-yellow-400 focus:ring focus:ring-yellow-200 transition-colors text-lg"
                                    />
                                </div>
                            </div>

                            <div>
                                <label className="block text-purple-600 text-lg font-medium mb-2">
                                    Create a Secret Password:
                                </label>
                                <div className="relative">
                                    <span className="absolute left-3 top-3 text-xl">🔐</span>
                                    <input
                                        type="password"
                                        name="password"
                                        value={formData.password}
                                        onChange={handleChange}
                                        required
                                        className="w-full pl-10 pr-3 py-3 rounded-xl border-2 border-blue-300 focus:border-yellow-400 focus:ring focus:ring-yellow-200 transition-colors text-lg"
                                    />
                                </div>
                            </div>

                            <button
                                type="submit"
                                className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-blue-500 to-purple-600 text-white text-lg font-bold shadow-lg hover:shadow-xl transform hover:-translate-y-1 transition-all duration-200"
                            >
                                Start My Journey! 🚀
                            </button>
                        </form>

                        <div className="mt-6 text-center">
                            <button
                                onClick={() => setActiveComponent("login")}
                                className="text-lg text-blue-600 hover:text-purple-600 transition-colors duration-200 flex items-center justify-center mx-auto"
                            >
                                <span className="mr-2">🌟</span>
                                Already on the journey? Login here!
                                <span className="ml-2">🌟</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default function AuthPageForBalvidya() {
    const [activeComponent, setActiveComponent] = useState("login");
    const [emailToPass, setEmailToPass] = useState("");
    return (
        <div>
            {activeComponent === "login" && <Login setActiveComponent={setActiveComponent} setEmailToPass={setEmailToPass} />}
            {activeComponent === "register" && <Register setActiveComponent={setActiveComponent} setEmailToPass={setEmailToPass} />}
            {activeComponent === "dashboard" && <BalVidyaDashboard setActiveComponent={setActiveComponent} email={emailToPass} />}
        </div>
    );
}