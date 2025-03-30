import React, { useState, useEffect } from 'react';
import { Link } from "react-router-dom";
export default function Navbar({ setActiveComponent }) {
    const [hoverEffect, setHoverEffect] = useState(null);
    const [particles, setParticles] = useState([]);

    // Generate divine particles on component mount
    useEffect(() => {
        const divineParticles = [];
        for (let i = 0; i < 40; i++) {
            divineParticles.push({
                top: Math.random() * 100 + '%',
                left: Math.random() * 100 + '%',
                size: Math.random() * 2 + 1 + 'px',
                animationDuration: Math.random() * 3 + 2 + 's'
            });
        }
        setParticles(divineParticles);
    }, []);

    return (
        <nav className="relative flex justify-between items-center p-6 bg-gray-950 z-10 overflow-hidden">
            {/* Ancient-meets-futuristic background */}
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0wIDBoNjB2NjBoLTYweiIvPjxwYXRoIGQ9Ik0zMCAzMCBMMTUgMTUgTDQ1IDE1IFoiIHN0cm9rZT0iIzBmMTYyYyIgc3Ryb2tlLXdpZHRoPSIwLjUiIGZpbGw9Im5vbmUiLz48cGF0aCBkPSJNMzAgMzAgTDE1IDQ1IEw0NSA0NSBaIiBzdHJva2U9IiMwZjE2MmMiIHN0cm9rZS13aWR0aD0iMC41IiBmaWxsPSJub25lIi8+PHBhdGggZD0iTTMwIDMwIEwxNSAxNSBMMTUgNDUgWiIgc3Ryb2tlPSIjMGYxNjJjIiBzdHJva2Utd2lkdGg9IjAuNSIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0zMCAzMCBMNDUgMTUgTDQ1IDQ1IFoiIHN0cm9rZT0iIzBmMTYyYyIgc3Ryb2tlLXdpZHRoPSIwLjUiIGZpbGw9Im5vbmUiLz48L2c+PC9zdmc+')] opacity-20"></div>

            {/* Ornate sacred border with modern twist */}
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-amber-600 via-teal-500 to-amber-600 opacity-80"></div>
            <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-amber-600 via-teal-500 to-amber-600 opacity-50"></div>

            {/* Sacred symbols in corners */}
            <div className="absolute top-2 left-2 w-12 h-12 border-t-2 border-l-2 border-amber-700/30 opacity-60"></div>
            <div className="absolute top-2 right-2 w-12 h-12 border-t-2 border-r-2 border-amber-700/30 opacity-60"></div>
            <div className="absolute bottom-2 left-2 w-12 h-12 border-b-2 border-l-2 border-amber-700/30 opacity-60"></div>
            <div className="absolute bottom-2 right-2 w-12 h-12 border-b-2 border-r-2 border-amber-700/30 opacity-60"></div>

            {/* Divine particles */}
            <div className="absolute inset-0 overflow-hidden">
                {particles.map((particle, index) => (
                    <div
                        key={index}
                        className="absolute rounded-full bg-amber-300 animate-pulse"
                        style={{
                            top: particle.top,
                            left: particle.left,
                            width: particle.size,
                            height: particle.size,
                            animationDuration: particle.animationDuration,
                            opacity: 0.2
                        }}
                    />
                ))}
            </div>

            {/* Logo with epic-meets-web3 styling */}
            <div className="flex items-center space-x-3 group relative z-100">
                <div className="relative">
                    <div className="w-12 h-12 flex items-center justify-center overflow-hidden shadow-lg group-hover:shadow-amber-700/30 transition-all duration-500 bg-gray-900 border border-amber-700/20">
                        {/* Chakra wheel design */}
                        <div className="absolute w-10 h-10 border border-amber-500/40 rounded-full"></div>
                        <div className="absolute w-8 h-8 border border-teal-500/30 rounded-full"></div>
                        <div className="absolute w-6 h-6 border border-amber-500/40 rounded-full"></div>

                        {/* Spokes */}
                        {[...Array(8)].map((_, i) => (
                            <div
                                key={i}
                                className="absolute w-10 h-0.5 bg-amber-600/30"
                                style={{ transform: `rotate(${i * 45}deg)` }}
                            ></div>
                        ))}

                        {/* Central divine light */}
                        <div className="absolute w-3 h-3 bg-teal-400 rounded-full animate-pulse"></div>
                    </div>
                    {/* Divine glow on hover */}
                    <div className="absolute -inset-1 bg-gradient-to-r from-amber-600/20 to-teal-600/20 rounded-full blur opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                </div>
                <h1 className="text-2xl font-bold relative text-white">
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-amber-400 to-amber-300 group-hover:from-amber-300 group-hover:to-amber-200 transition-all duration-500">Chakra</span>
                    <span className="text-teal-500 group-hover:text-teal-400 transition-all duration-500">Vidya</span>

                    {/* Sanskrit-inspired decorative elements */}
                    <div className="absolute -bottom-2 left-0 w-full h-0.5 bg-amber-700/30"></div>
                    <div className="absolute -top-2 left-1/4 w-1/2 h-0.5 bg-amber-700/30"></div>
                </h1>
            </div>

            {/* Epic-themed navigation buttons with modern tech styling */}
            <div className="flex space-x-1 md:space-x-4 relative z-100">
                {[
                    {
                        icon: "🧒", name: "BalVidya", description: "From Kindergarden to Class 5th. 🏷️ Where Learning Begins with Fun & Wonder!", path: "/balvidya"
                    },
                    { icon: "📚", name: "Madhyamik", description: "From Grade 6th to Grade 10th 🏷️ Building Strong Foundations for a Bright Future!", path: "/madhyamik" },
                    {
                        icon: "🔮", name: "UchhaShiksha", description: "For Grade 11th and 12th 🏷️ Sharpening Minds for the Next Big Leap!", path: "/uchhashiksha"
                    },
                    { icon: "✨", name: "VisheshVidyarthi", description: "For College & Postgraduate 🏷️ Empowering Scholars to Innovate & Lead!", path: "/visheshvidyarthi" }

                ].map((item, index) => (
                    <div
                        key={index}
                        className="relative group"
                        onMouseEnter={() => setHoverEffect(item.name)}
                        onMouseLeave={() => setHoverEffect(null)}
                        onClick={() => setActiveComponent(item.name)}
                    >

                        <div className="relative z-150">
                            <button className="px-3 py-2 text-amber-200/80 hover:text-amber-300 transition-all duration-300 flex items-center space-x-2 relative z-40 backdrop-blur-sm border border-amber-900/30 hover:border-amber-700/50 rounded-md bg-gray-900/80">
                                <span className="text-lg">{item.icon}</span>
                                <span className="hidden md:block font-medium">{item.name}</span>

                                {/* Sacred glow effect */}
                                <div className="absolute inset-0 bg-amber-900/5 opacity-0 group-hover:opacity-100 transition-all duration-500 rounded-md"></div>

                                {/* Animated ornate border */}
                                <div className="absolute bottom-0 left-0 w-0 h-0.5 bg-gradient-to-r from-amber-500 to-teal-500 group-hover:w-full transition-all duration-700"></div>
                            </button>
                        </div>

                        {/* Epic scroll modal overlay */}
                        <div className={`fixed inset-0 bg-gray-950 bg-opacity-95 backdrop-blur-md opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none group-hover:pointer-events-auto z-10 ${hoverEffect === item.name ? 'overlay-active' : ''}`}>
                            {/* Ancient scroll background with tech elements */}
                            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0wIDBoNjB2NjBoLTYweiIvPjxwYXRoIGQ9Ik0zMCAzMCBMMTUgMTUgTDQ1IDE1IFoiIHN0cm9rZT0iIzJhMWUwZCIgc3Ryb2tlLXdpZHRoPSIwLjUiIGZpbGw9Im5vbmUiLz48cGF0aCBkPSJNMzAgMzAgTDE1IDQ1IEw0NSA0NSBaIiBzdHJva2U9IiMyYTFlMGQiIHN0cm9rZS13aWR0aD0iMC41IiBmaWxsPSJub25lIi8+PC9nPjwvc3ZnPg==')] opacity-10"></div>

                            {/* Divine light particles */}
                            <div className="absolute inset-0 overflow-hidden">
                                {[...Array(15)].map((_, i) => (
                                    <div
                                        key={i}
                                        className="absolute rounded-full bg-amber-500 opacity-20 animate-float-slow"
                                        style={{
                                            width: `${Math.random() * 20 + 5}px`,
                                            height: `${Math.random() * 20 + 5}px`,
                                            top: `${Math.random() * 100}%`,
                                            left: `${Math.random() * 100}%`,
                                            animationDuration: `${Math.random() * 10 + 15}s`,
                                            animationDelay: `${Math.random() * 5}s`
                                        }}
                                    ></div>
                                ))}
                            </div>

                            <div className="absolute inset-0 flex items-center justify-center">
                                <div className="max-w-lg text-center p-8 relative">
                                    {/* Ornate scroll frame */}
                                    <div className="absolute inset-0 border border-amber-800/20 rounded-md">
                                        {/* Corner ornaments */}
                                        <div className="absolute -top-2 -left-2 w-6 h-6 border-t-2 border-l-2 border-amber-600/40"></div>
                                        <div className="absolute -top-2 -right-2 w-6 h-6 border-t-2 border-r-2 border-amber-600/40"></div>
                                        <div className="absolute -bottom-2 -left-2 w-6 h-6 border-b-2 border-l-2 border-amber-600/40"></div>
                                        <div className="absolute -bottom-2 -right-2 w-6 h-6 border-b-2 border-r-2 border-amber-600/40"></div>
                                    </div>

                                    <div className="text-5xl mb-6 transform transition-all duration-700 group-hover:scale-110 animate-float-slow">{item.icon}</div>
                                    <h2 className="text-3xl font-bold text-amber-500 mb-4 relative">
                                        {item.name}
                                        {/* Sanskrit-inspired underline */}
                                        <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-0 h-0.5 bg-gradient-to-r from-transparent via-amber-500 to-transparent group-hover:w-48 transition-all duration-1000 delay-300"></div>
                                    </h2>
                                    <p className="text-lg text-amber-100/80 mb-8 opacity-0 transform translate-y-4 group-hover:opacity-100 group-hover:translate-y-0 transition-all duration-700 delay-300">{item.description}</p>

                                    {/* Sacred yantra visualization with tech elements */}
                                    <div className="w-32 h-32 mx-auto relative opacity-80 group-hover:opacity-100 transition-all duration-1000">
                                        <div className="absolute inset-0 border border-amber-600/30 rounded-md opacity-60 animate-pulse"></div>

                                        {/* Triangles - representing both sacred geometry and tech elements */}
                                        <div className="absolute inset-4 w-24 h-24 opacity-30 animate-spin" style={{ animationDuration: '20s' }}>
                                            <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-[12px] border-r-[12px] border-b-[20px] border-l-transparent border-r-transparent border-b-amber-600/30"></div>
                                            <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 rotate-180 w-0 h-0 border-l-[12px] border-r-[12px] border-b-[20px] border-l-transparent border-r-transparent border-b-amber-600/30"></div>
                                            <div className="absolute top-1/2 left-0 transform -translate-y-1/2 rotate-270 w-0 h-0 border-l-[12px] border-r-[12px] border-b-[20px] border-l-transparent border-r-transparent border-b-amber-600/30"></div>
                                            <div className="absolute top-1/2 right-0 transform -translate-y-1/2 rotate-90 w-0 h-0 border-l-[12px] border-r-[12px] border-b-[20px] border-l-transparent border-r-transparent border-b-amber-600/30"></div>
                                        </div>

                                        <div className="absolute inset-10 border border-teal-400/20 rounded-full opacity-50 animate-spin" style={{ animationDuration: '10s' }}></div>
                                        <div className="absolute inset-0 bg-gradient-to-br from-amber-600/5 to-teal-600/5 rounded-md animate-pulse" style={{ animationDuration: '4s' }}></div>
                                    </div>
                                </div>
                            </div>

                        </div>
                    </div>
                ))}
            </div>




            {/* Animation styles */}
            <style jsx>{`
                @keyframes float {
                    0%, 100% { transform: translateY(0) translateX(0); }
                    50% { transform: translateY(-10px) translateX(5px); }
                }
                @keyframes float-slow {
                    0%, 100% { transform: translateY(0) translateX(0); }
                    25% { transform: translateY(-15px) translateX(10px); }
                    75% { transform: translateY(15px) translateX(-10px); }
                }
                .animate-float {
                    animation: float 3s ease-in-out infinite;
                }
                .animate-float-slow {
                    animation: float-slow 8s ease-in-out infinite;
                }
                .overlay-active {
                    animation: overlay-pulse 5s ease-in-out infinite;
                }
                @keyframes overlay-pulse {
                    0%, 100% { background-color: rgba(3, 7, 18, 0.95); }
                    50% { background-color: rgba(30, 25, 14, 0.95); }
                }
            `}</style>
        </nav>
    );
}