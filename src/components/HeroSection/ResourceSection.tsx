import React, { useState, useEffect } from "react";

const ResourceSection = ({ title = "Recommended Reading", resources = [] }) => {
    const [particles, setParticles] = useState([]);
    const [glowIntensity, setGlowIntensity] = useState(30);

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

        // Pulsating glow effect
        const glowInterval = setInterval(() => {
            setGlowIntensity(prev => {
                const newValue = prev + (Math.random() * 2 - 1);
                return Math.max(20, Math.min(40, newValue));
            });
        }, 1000);

        return () => clearInterval(glowInterval);
    }, []);

    return (
        <section className="relative py-12 px-6 bg-gray-950 overflow-hidden">
            {/* Ancient-meets-futuristic background */}
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0wIDBoNjB2NjBoLTYweiIvPjxwYXRoIGQ9Ik0zMCAzMCBMMTUgMTUgTDQ1IDE1IFoiIHN0cm9rZT0iIzBmMTYyYyIgc3Ryb2tlLXdpZHRoPSIwLjUiIGZpbGw9Im5vbmUiLz48cGF0aCBkPSJNMzAgMzAgTDE1IDQ1IEw0NSA0NSBaIiBzdHJva2U9IiMwZjE2MmMiIHN0cm9rZS13aWR0aD0iMC41IiBmaWxsPSJub25lIi8+PHBhdGggZD0iTTMwIDMwIEwxNSAxNSBMMTUgNDUgWiIgc3Ryb2tlPSIjMGYxNjJjIiBzdHJva2Utd2lkdGg9IjAuNSIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0zMCAzMCBMNDUgMTUgTDQ1IDQ1IFoiIHN0cm9rZT0iIzBmMTYyYyIgc3Ryb2tlLXdpZHRoPSIwLjUiIGZpbGw9Im5vbmUiLz48L2c+PC9zdmc+')] opacity-20"></div>

            {/* Animated energy field */}
            <div className="absolute inset-0 bg-gradient-to-br from-purple-900/5 via-transparent to-teal-900/5 animate-pulse"></div>

            {/* Ornate sacred border with modern twist */}
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-purple-700 via-teal-500 to-amber-600 opacity-80"></div>
            <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-amber-600 via-teal-500 to-purple-700 opacity-50"></div>

            {/* Sacred symbols in corners */}
            <div className="absolute top-4 left-4 w-20 h-20 border-t-2 border-l-2 border-amber-700/30 opacity-60 animate-float"></div>
            <div className="absolute top-4 right-4 w-20 h-20 border-t-2 border-r-2 border-teal-700/30 opacity-60 animate-float-slow"></div>
            <div className="absolute bottom-4 left-4 w-20 h-20 border-b-2 border-l-2 border-teal-700/30 opacity-60 animate-float-slow"></div>
            <div className="absolute bottom-4 right-4 w-20 h-20 border-b-2 border-r-2 border-amber-700/30 opacity-60 animate-float"></div>

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
            <div className="relative z-10 mb-12 text-center">
                <h2 className="text-4xl font-bold relative inline-block">
                    <span className="text-transparent bg-clip-text bg-gradient-to-br from-amber-400 via-amber-300 to-teal-400">{title.split(' ')[0]} </span>
                    <span className="text-transparent bg-clip-text bg-gradient-to-br from-teal-400 to-purple-500">{title.split(' ').slice(1).join(' ')}</span>

                    {/* Mystical symbol above title */}
                    <div className="absolute -top-8 left-1/2 transform -translate-x-1/2">
                        <div className="w-10 h-10 mx-auto">
                            <svg viewBox="0 0 100 100" className="w-full h-full">
                                <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(217, 119, 6, 0.2)" strokeWidth="2" />
                                <path d="M50,5 L50,95 M5,50 L95,50 M26,26 L74,74 M26,74 L74,26" stroke="rgba(217, 119, 6, 0.2)" strokeWidth="1" />
                                <circle cx="50" cy="50" r="10" fill="none" stroke="rgba(13, 148, 136, 0.3)" strokeWidth="1" className="animate-spin-slow" style={{ transformOrigin: "center" }} />
                            </svg>
                        </div>
                    </div>

                    {/* Sanskrit-inspired decorative elements */}
                    <div className="absolute -bottom-2 left-0 w-full h-0.5 bg-gradient-to-r from-amber-700/30 via-teal-700/30 to-amber-700/30"></div>
                    <div className="absolute -top-2 left-1/4 w-1/2 h-0.5 bg-gradient-to-r from-teal-700/30 via-amber-700/30 to-teal-700/30"></div>
                </h2>
            </div>

            {/* Introduction text */}
            <div className="relative z-10 max-w-4xl mx-auto mb-8 text-center">
                <p className="text-amber-100/80 text-lg">Explore these sacred texts and teachings to deepen your knowledge of Indian culture and great thinkers:</p>
            </div>

            {/* Resources List */}
            <div className="relative z-10 max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-6">
                {resources.map((resource, index) => (
                    <ResourceCard
                        key={index}
                        title={resource.title}
                        description={resource.description}
                        link={resource.link}
                        glowIntensity={glowIntensity}
                        delay={index * 0.2}
                    />
                ))}
            </div>

            {/* Animation styles */}
            <style jsx>{`
                @keyframes float {
                    0%, 100% { transform: translateY(0) translateX(0) rotate(0deg); }
                    50% { transform: translateY(-10px) translateX(5px) rotate(1deg); }
                }
                @keyframes float-slow {
                    0%, 100% { transform: translateY(0) translateX(0) rotate(0deg); }
                    25% { transform: translateY(-15px) translateX(10px) rotate(-1deg); }
                    75% { transform: translateY(15px) translateX(-10px) rotate(1deg); }
                }
                @keyframes spin-slow {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
                .animate-float {
                    animation: float 3s ease-in-out infinite;
                }
                .animate-float-slow {
                    animation: float-slow 8s ease-in-out infinite;
                }
                .animate-spin-slow {
                    animation: spin-slow 12s linear infinite;
                }
            `}</style>
        </section>
    );
};

// Resource Card Component
const ResourceCard = ({ title, description, link, glowIntensity, delay = 0 }) => {
    return (
        <div
            className="bg-gray-900/80 border border-amber-900/30 hover:border-amber-700/50 transition-all duration-700 p-6 rounded-lg transform hover:scale-105 hover:shadow-xl hover:z-20"
            style={{
                boxShadow: `0 0 ${glowIntensity}px 5px rgba(217, 119, 6, 0.1)`,
                backdropFilter: "blur(12px)",
                opacity: 0,
                animation: "fadeIn 0.8s ease-out forwards",
                animationDelay: `${delay}s`
            }}
        >
            {/* Animated Content Aura Effect */}
            <div className="absolute -inset-1 bg-gradient-to-r from-amber-600/10 via-teal-600/10 to-purple-600/10 rounded-lg blur opacity-30 animate-pulse"></div>

            {/* Enhanced Corner ornaments with subtle animation */}
            <div className="absolute top-1 left-1 w-6 h-6 border-t border-l border-amber-600/40 animate-pulse"></div>
            <div className="absolute top-1 right-1 w-6 h-6 border-t border-r border-teal-600/40 animate-pulse"></div>
            <div className="absolute bottom-1 left-1 w-6 h-6 border-b border-l border-teal-600/40 animate-pulse"></div>
            <div className="absolute bottom-1 right-1 w-6 h-6 border-b border-r border-amber-600/40 animate-pulse"></div>

            {/* Card Title */}
            <div className="relative mb-4">
                <h3 className="text-xl font-bold relative inline-block">
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-amber-400 to-teal-400">
                        {title}
                    </span>
                    <div className="absolute -bottom-1 left-0 w-full h-px bg-gradient-to-r from-transparent via-amber-700/40 to-transparent"></div>
                </h3>
            </div>

            {/* Card Content */}
            <div className="relative z-10 text-amber-100/80 leading-relaxed mb-4">
                {description}
            </div>

            {/* Mystical Read More Link */}
            <div className="relative z-10 mt-4">
                <a
                    href={link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-block text-sm text-teal-400 hover:text-amber-400 transition-colors duration-300"
                >
                    <span className="relative group">
                        <span>Journey into Knowledge</span>
                        <span className="absolute -bottom-1 left-0 w-0 h-px bg-amber-400 group-hover:w-full transition-all duration-300"></span>
                    </span>
                    <svg className="w-4 h-4 inline-block ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path>
                    </svg>
                </a>
            </div>

            {/* Mystical symbol at bottom */}
            <div className="mt-4 flex justify-center">
                <svg viewBox="0 0 100 30" className="w-16 h-4 opacity-30">
                    <path d="M0,15 L40,15 M60,15 L100,15" stroke="rgba(217, 119, 6, 0.4)" strokeWidth="1" />
                    <circle cx="50" cy="15" r="5" fill="none" stroke="rgba(13, 148, 136, 0.4)" strokeWidth="1" />
                    <path d="M45,15 L55,15" stroke="rgba(217, 119, 6, 0.4)" strokeWidth="1" />
                </svg>
            </div>

            <style jsx>{`
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
            `}</style>
        </div>
    );
};

export default ResourceSection;