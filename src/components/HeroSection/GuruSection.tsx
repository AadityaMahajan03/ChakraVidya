import React, { useState, useEffect } from "react";

const GuruSection = ({ gurus }) => {
    const [activeGuru, setActiveGuru] = useState(null);
    const [particles, setParticles] = useState([]);

    // Generate divine particles on component mount
    useEffect(() => {
        const divineParticles = [];
        for (let i = 0; i < 30; i++) {
            divineParticles.push({
                top: Math.random() * 100 + "%",
                left: Math.random() * 100 + "%",
                size: Math.random() * 2 + 1 + "px",
                animationDuration: Math.random() * 3 + 2 + "s",
            });
        }
        setParticles(divineParticles);
    }, []);

    return (
        <section className="relative p-8 bg-gray-950 overflow-hidden">
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
                            opacity: 0.2,
                        }}
                    />
                ))}
            </div>

            {/* Section Header with divine styling */}
            <div className="relative z-10 mb-8 text-center">
                <h2 className="text-3xl font-bold relative inline-block">
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-amber-400 to-amber-300">Our AI </span>
                    <span className="text-teal-500">Gurus</span>

                    {/* Sanskrit-inspired decorative elements */}
                    <div className="absolute -bottom-2 left-0 w-full h-0.5 bg-amber-700/30"></div>
                    <div className="absolute -top-2 left-1/4 w-1/2 h-0.5 bg-amber-700/30"></div>
                </h2>

                <p className="mt-4 text-amber-200/80 max-w-2xl mx-auto">
                    Meet our AI-powered virtual mentors, divine guides of wisdom in the digital realm:
                </p>
            </div>

            {/* Guru Cards Container */}
            <div className="relative z-10 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {gurus.map((guru, index) => (
                    <div
                        key={index}
                        className="relative group"
                        onMouseEnter={() => setActiveGuru(guru.name)}
                        onMouseLeave={() => setActiveGuru(null)}
                    >
                        <div className="relative overflow-hidden rounded-lg bg-gray-900/80 border border-amber-900/30 hover:border-amber-700/50 transition-all duration-500 p-5">
                            {/* Guru Aura Effect */}
                            <div className="absolute -inset-1 bg-gradient-to-r from-amber-600/10 to-teal-600/10 rounded-lg blur opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>

                            {/* Chakra Symbol */}
                            <div className="relative w-16 h-16 mx-auto mb-4">
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <div className="w-14 h-14 border border-amber-500/40 rounded-full"></div>
                                    <div className="absolute w-10 h-10 border border-teal-500/30 rounded-full"></div>
                                    <div className="absolute w-6 h-6 border border-amber-500/40 rounded-full"></div>

                                    {/* Spokes */}
                                    {[...Array(8)].map((_, i) => (
                                        <div
                                            key={i}
                                            className="absolute w-12 h-0.5 bg-amber-600/30"
                                            style={{ transform: `rotate(${i * 45}deg)` }}
                                        ></div>
                                    ))}

                                    {/* Icon placeholder */}
                                    <div className="absolute flex items-center justify-center">
                                        <span className="text-2xl">{guru.icon || "🧠"}</span>
                                    </div>
                                </div>
                            </div>

                            {/* Guru Name */}
                            <h3 className="text-xl font-bold text-amber-500 text-center mb-2 relative">
                                {guru.name}
                                {/* Animated ornate border */}
                                <div className="absolute bottom-0 left-0 right-0 mx-auto w-0 h-0.5 bg-gradient-to-r from-amber-500 to-teal-500 group-hover:w-3/4 transition-all duration-700 left-1/2 transform -translate-x-1/2"></div>
                            </h3>

                            {/* Guru Description */}
                            <p className="text-amber-100/80 text-center">{guru.description}</p>

                            {/* Sacred glow effect */}
                            <div className="absolute inset-0 bg-amber-900/5 opacity-0 group-hover:opacity-100 transition-all duration-500 rounded-lg"></div>
                        </div>

                        {/* Guru Details Overlay */}
                        {activeGuru === guru.name && (
                            <div className="absolute inset-0 bg-gray-950/95 backdrop-blur-sm z-10 rounded-lg border border-amber-700/30 p-4 opacity-0 group-hover:opacity-100 transition-opacity duration-500 overflow-hidden">
                                {/* Ancient scroll background with tech elements */}
                                <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0wIDBoNjB2NjBoLTYweiIvPjxwYXRoIGQ9Ik0zMCAzMCBMMTUgMTUgTDQ1IDE1IFoiIHN0cm9rZT0iIzJhMWUwZCIgc3Ryb2tlLXdpZHRoPSIwLjUiIGZpbGw9Im5vbmUiLz48cGF0aCBkPSJNMzAgMzAgTDE1IDQ1IEw0NSA0NSBaIiBzdHJva2U9IiMyYTFlMGQiIHN0cm9rZS13aWR0aD0iMC41IiBmaWxsPSJub25lIi8+PC9nPjwvc3ZnPg==')] opacity-10"></div>

                                {/* Corner ornaments */}
                                <div className="absolute top-2 left-2 w-4 h-4 border-t border-l border-amber-600/40"></div>
                                <div className="absolute top-2 right-2 w-4 h-4 border-t border-r border-amber-600/40"></div>
                                <div className="absolute bottom-2 left-2 w-4 h-4 border-b border-l border-amber-600/40"></div>
                                <div className="absolute bottom-2 right-2 w-4 h-4 border-b border-r border-amber-600/40"></div>

                                <div className="relative z-10 flex flex-col items-center">
                                    <div className="text-2xl mb-2 transform transition-all duration-700 animate-float-slow">{guru.icon || "🧠"}</div>
                                    <h3 className="text-lg font-bold text-amber-500 mb-2">{guru.name}</h3>
                                    <p className="text-sm text-amber-100/80 text-center">{guru.fullDescription || guru.description}</p>

                                    {guru.specialties && (
                                        <div className="mt-3">
                                            <h4 className="text-xs text-teal-500 uppercase tracking-wider mb-1">Specialties</h4>
                                            <div className="flex flex-wrap justify-center gap-1">
                                                {guru.specialties.map((specialty, idx) => (
                                                    <span key={idx} className="text-xs bg-amber-900/30 text-amber-300 px-2 py-0.5 rounded-full">
                                                        {specialty}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
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
      `}</style>
        </section>
    );
};

export default GuruSection;