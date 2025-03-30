import React, { useState, useEffect } from "react";

const ListSection = ({ title, items }) => {
    const [activeCard, setActiveCard] = useState(null);
    const [particles, setParticles] = useState([]);

    // Generate divine particles on component mount
    useEffect(() => {
        const divineParticles = [];
        for (let i = 0; i < 25; i++) {
            divineParticles.push({
                top: Math.random() * 100 + "%",
                left: Math.random() * 100 + "%",
                size: Math.random() * 2 + 1 + "px",
                animationDuration: Math.random() * 3 + 2 + "s",
            });
        }
        setParticles(divineParticles);
    }, []);

    // Generate icons/symbols for each card
    const symbols = ["✨", "🔮", "⚡", "🌟", "🏆", "💫", "🌠", "🧿"];

    return (
        <section className="relative py-12 px-6 bg-gray-950 overflow-hidden">
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
            <div className="relative z-10 mb-10 text-center">
                <h2 className="text-3xl font-bold relative inline-block">
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-amber-400 to-amber-300">{title}</span>

                    {/* Sanskrit-inspired decorative elements */}
                    <div className="absolute -bottom-2 left-0 w-full h-0.5 bg-amber-700/30"></div>
                    <div className="absolute -top-2 left-1/4 w-1/2 h-0.5 bg-amber-700/30"></div>
                </h2>

                {/* Ornate divider */}
                <div className="mt-6 flex justify-center">
                    <div className="relative w-32 h-4">
                        <div className="absolute top-1/2 left-0 w-10 h-0.5 bg-amber-700/40 transform -translate-y-1/2"></div>
                        <div className="absolute top-1/2 right-0 w-10 h-0.5 bg-amber-700/40 transform -translate-y-1/2"></div>
                        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                            <div className="w-4 h-4 border border-amber-700/40 rotate-45"></div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Sacred Cards Grid */}
            <div className="relative z-10 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
                {items.map((item, index) => (
                    <div
                        key={index}
                        className="relative group"
                        onMouseEnter={() => setActiveCard(index)}
                        onMouseLeave={() => setActiveCard(null)}
                    >
                        {/* Card with 3D effect */}
                        <div className="relative h-full bg-gradient-to-br from-gray-900 to-gray-950 rounded-lg overflow-hidden border border-amber-900/30 group-hover:border-amber-700/50 transition-all duration-500 shadow-lg group-hover:shadow-amber-900/20 transform group-hover:-translate-y-1 group-hover:scale-105">
                            {/* Mystical background pattern */}
                            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0wIDBoNjB2NjBoLTYweiIvPjxwYXRoIGQ9Ik0zMCAzMCBMMTUgMTUgTDQ1IDE1IFoiIHN0cm9rZT0iIzBmMTYyYyIgc3Ryb2tlLXdpZHRoPSIwLjUiIGZpbGw9Im5vbmUiLz48cGF0aCBkPSJNMzAgMzAgTDE1IDQ1IEw0NSA0NSBaIiBzdHJva2U9IiMwZjE2MmMiIHN0cm9rZS13aWR0aD0iMC41IiBmaWxsPSJub25lIi8+PC9nPjwvc3ZnPg==')] opacity-10"></div>

                            {/* Card content */}
                            <div className="relative p-6 h-full flex flex-col">
                                {/* Sacred symbol header */}
                                <div className="mb-4 flex items-center justify-center">
                                    <div className="relative">
                                        {/* Sacred yantra background */}
                                        <div className="w-16 h-16 relative">
                                            {/* Rotating geometric shapes */}
                                            <div className="absolute inset-0 opacity-60 animate-spin-slow">
                                                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-[8px] border-r-[8px] border-b-[14px] border-l-transparent border-r-transparent border-b-amber-600/30"></div>
                                                <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 rotate-180 w-0 h-0 border-l-[8px] border-r-[8px] border-b-[14px] border-l-transparent border-r-transparent border-b-amber-600/30"></div>
                                                <div className="absolute top-1/2 left-0 transform -translate-y-1/2 rotate-270 w-0 h-0 border-l-[8px] border-r-[8px] border-b-[14px] border-l-transparent border-r-transparent border-b-amber-600/30"></div>
                                                <div className="absolute top-1/2 right-0 transform -translate-y-1/2 rotate-90 w-0 h-0 border-l-[8px] border-r-[8px] border-b-[14px] border-l-transparent border-r-transparent border-b-amber-600/30"></div>
                                            </div>
                                            <div className="absolute inset-2 border border-amber-500/30 rounded-full opacity-70"></div>
                                            <div className="absolute inset-4 border border-teal-500/20 rounded-full opacity-70"></div>

                                            {/* Central symbol */}
                                            <div className="absolute inset-0 flex items-center justify-center">
                                                <span className="text-2xl group-hover:animate-float-slow">
                                                    {symbols[index % symbols.length]}
                                                </span>
                                            </div>
                                        </div>

                                        {/* Divine glow on hover */}
                                        <div className="absolute -inset-2 bg-gradient-to-r from-amber-600/10 to-teal-600/10 rounded-full blur opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                                    </div>
                                </div>

                                {/* Card text */}
                                <div className="text-center mb-4 flex-grow">
                                    <p className="text-amber-200/90 group-hover:text-amber-300 transition-colors duration-300 font-medium">
                                        {item}
                                    </p>
                                </div>

                                {/* Sacred energy bar */}
                                <div className="relative h-1 w-full bg-gray-800 rounded overflow-hidden">
                                    <div
                                        className="h-full bg-gradient-to-r from-amber-600 to-teal-500 transition-all duration-1000 ease-in-out"
                                        style={{
                                            width: activeCard === index ? '100%' : '30%',
                                        }}
                                    ></div>
                                </div>

                                {/* Card corner ornaments */}
                                <div className="absolute top-2 left-2 w-4 h-4 border-t border-l border-amber-600/20"></div>
                                <div className="absolute top-2 right-2 w-4 h-4 border-t border-r border-amber-600/20"></div>
                                <div className="absolute bottom-2 left-2 w-4 h-4 border-b border-l border-amber-600/20"></div>
                                <div className="absolute bottom-2 right-2 w-4 h-4 border-b border-r border-amber-600/20"></div>
                            </div>

                            {/* Animated bottom border */}
                            <div
                                className="absolute bottom-0 left-0 h-0.5 bg-gradient-to-r from-amber-500 to-teal-500 transition-all duration-700 ease-in-out"
                                style={{ width: activeCard === index ? '100%' : '0%' }}
                            ></div>
                        </div>

                        {/* Floating particle effects on hover */}
                        {activeCard === index && (
                            <>
                                {[...Array(5)].map((_, i) => (
                                    <div
                                        key={i}
                                        className="absolute rounded-full bg-amber-400/30 animate-float-particle"
                                        style={{
                                            width: `${Math.random() * 6 + 2}px`,
                                            height: `${Math.random() * 6 + 2}px`,
                                            top: `${Math.random() * 100}%`,
                                            left: `${Math.random() * 100}%`,
                                            animationDuration: `${Math.random() * 2 + 1}s`,
                                            animationDelay: `${Math.random() * 0.5}s`
                                        }}
                                    />
                                ))}
                            </>
                        )}
                    </div>
                ))}
            </div>

            {/* Animation styles */}
            <style jsx>{`
        @keyframes float-slow {
          0%, 100% { transform: translateY(0) translateX(0) rotate(0deg); }
          25% { transform: translateY(-5px) translateX(3px) rotate(5deg); }
          75% { transform: translateY(5px) translateX(-3px) rotate(-5deg); }
        }
        .animate-float-slow {
          animation: float-slow 6s ease-in-out infinite;
        }
        @keyframes spin-slow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .animate-spin-slow {
          animation: spin-slow 20s linear infinite;
        }
        @keyframes float-particle {
          0% { transform: translateY(0) translateX(0); opacity: 0; }
          50% { opacity: 1; }
          100% { transform: translateY(-20px) translateX(10px); opacity: 0; }
        }
        .animate-float-particle {
          animation: float-particle 2s ease-out forwards;
        }
      `}</style>
        </section>
    );
};

export default ListSection;