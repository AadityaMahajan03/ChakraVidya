import React, { useState, useEffect } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

const CarousalOfHome = ({ slides = [] }) => {
    const [currentIndex, setCurrentIndex] = useState(0);
    const [isTransitioning, setIsTransitioning] = useState(false);
    const [planets, setPlanets] = useState([]);
    const [stars, setStars] = useState([]);
    const [comets, setComets] = useState([]);

    // Default slides with original Mahabharata content
    const defaultSlides = [
        {
            title: "Divine Knowledge",
            subtitle: "Wisdom of the Bhagavad Gita",
            description: "Ancient wisdom from Lord Krishna that transcends time and illuminates the path of dharma."
        },
        {
            title: "Epic Battle",
            subtitle: "Kurukshetra War",
            description: "The legendary 18-day battle that changed the course of history and revealed cosmic truths."
        },
        {
            title: "Celestial Weapons",
            subtitle: "Divine Astras",
            description: "Powerful weapons of the gods that commanded the elements and cosmic forces."
        },
        {
            title: "Royal Legacy",
            subtitle: "Hastinapur Dynasty",
            description: "The noble lineage of kings and the complex web of duty, honor, and sacrifice."
        }
    ];

    // Merge provided slides with default slides
    const carouselSlides = slides.length > 0 ? slides : defaultSlides;

    // Generate cosmic elements for the solar system theme
    useEffect(() => {
        // Stars for cosmic background
        const starField = [];
        for (let i = 0; i < 300; i++) {
            starField.push({
                top: Math.random() * 100 + "%",
                left: Math.random() * 100 + "%",
                size: Math.random() * 2.5 + 0.5,
                opacity: Math.random() * 0.9 + 0.1,
                animationDuration: Math.random() * 4 + 2,
                animationDelay: Math.random() * 5,
                color: i % 20 === 0 ? '#8BE9FD' : i % 15 === 0 ? '#FFB86C' : '#F8F8F2'
            });
        }
        setStars(starField);

        // Planets/asteroids in background
        const spaceObjects = [];
        const planetColors = [
            'rgba(210, 180, 140, 0.6)', // Mercury (tan)
            'rgba(255, 190, 120, 0.6)', // Venus (light orange)
            'rgba(70, 130, 180, 0.6)',  // Earth (steel blue)
            'rgba(205, 92, 92, 0.6)',   // Mars (indian red)
            'rgba(244, 164, 96, 0.6)',  // Jupiter (sandy brown)
            'rgba(210, 180, 140, 0.6)', // Saturn (tan)
            'rgba(173, 216, 230, 0.6)', // Uranus (light blue)
            'rgba(65, 105, 225, 0.6)'   // Neptune (royal blue)
        ];

        for (let i = 0; i < 8; i++) {
            spaceObjects.push({
                top: 30 + Math.random() * 40 + "%",
                left: 10 + Math.random() * 80 + "%",
                size: 6 + Math.random() * 25,
                color: planetColors[i],
                glow: `0 0 ${10 + Math.random() * 20}px ${planetColors[i].replace("0.6", "0.3")}`,
                orbitDuration: (i + 1) * 120,
                hasRings: i === 5, // Saturn has rings
                rotationSpeed: 5 + Math.random() * 10
            });
        }
        setPlanets(spaceObjects);

        // Generate comets
        const cometObjects = [];
        for (let i = 0; i < 5; i++) {
            cometObjects.push({
                top: Math.random() * 60 + 20 + "%",
                left: "-10%",
                size: 2 + Math.random() * 3,
                speed: 15 + Math.random() * 25,
                delay: i * 7,
                tailLength: 80 + Math.random() * 150
            });
        }
        setComets(cometObjects);

        // Auto advance carousel
        const interval = setInterval(() => {
            nextSlide();
        }, 6000);

        return () => clearInterval(interval);
    }, []);

    const nextSlide = () => {
        if (!isTransitioning) {
            setIsTransitioning(true);
            setCurrentIndex((prevIndex) => (prevIndex === carouselSlides.length - 1 ? 0 : prevIndex + 1));
            setTimeout(() => setIsTransitioning(false), 500);
        }
    };

    const prevSlide = () => {
        if (!isTransitioning) {
            setIsTransitioning(true);
            setCurrentIndex((prevIndex) => (prevIndex === 0 ? carouselSlides.length - 1 : prevIndex - 1));
            setTimeout(() => setIsTransitioning(false), 500);
        }
    };

    const goToSlide = (index) => {
        if (!isTransitioning && index !== currentIndex) {
            setIsTransitioning(true);
            setCurrentIndex(index);
            setTimeout(() => setIsTransitioning(false), 500);
        }
    };

    // Get planet theme based on slide index
    const getPlanetTheme = (index) => {
        const themes = [
            { primary: "from-amber-900/30 via-orange-800/20 to-yellow-900/30", accent: "amber" },    // Mercury/Sun theme
            { primary: "from-yellow-800/30 via-orange-700/20 to-amber-800/30", accent: "yellow" },   // Venus theme
            { primary: "from-blue-900/30 via-cyan-800/20 to-teal-900/30", accent: "blue" },          // Earth theme
            { primary: "from-red-900/30 via-rose-800/20 to-orange-900/30", accent: "red" }           // Mars theme
        ];

        return themes[index % themes.length];
    };

    return (
        <div className="relative inset-0 w-full h-screen overflow-hidden bg-gray-950">
            {/* Deep Space Background */}
            <div className="absolute inset-0 bg-gradient-to-b from-gray-950 via-indigo-950/20 to-violet-950/20 overflow-hidden">
                {/* Stars */}
                {stars.map((star, index) => (
                    <div
                        key={`star-${index}`}
                        className="absolute rounded-full animate-pulse"
                        style={{
                            top: star.top,
                            left: star.left,
                            width: `${star.size}px`,
                            height: `${star.size}px`,
                            backgroundColor: star.color,
                            boxShadow: `0 0 ${star.size * 1.5}px rgba(255, 255, 255, ${star.opacity})`,
                            opacity: star.opacity,
                            animationDuration: `${star.animationDuration}s`,
                            animationDelay: `${star.animationDelay}s`,
                        }}
                    />
                ))}

                {/* Planets */}
                {planets.map((planet, index) => (
                    <div
                        key={`planet-${index}`}
                        className="absolute rounded-full"
                        style={{
                            top: planet.top,
                            left: planet.left,
                            width: `${planet.size}px`,
                            height: `${planet.size}px`,
                            backgroundColor: planet.color,
                            boxShadow: planet.glow,
                            animation: `orbit${index} ${planet.orbitDuration}s linear infinite, rotate ${planet.rotationSpeed}s linear infinite`
                        }}
                    >
                        {planet.hasRings && (
                            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-full h-1/3 rounded-full border-2 border-yellow-500/30 rotate-12"></div>
                        )}
                    </div>
                ))}

                {/* Comets */}
                {comets.map((comet, index) => (
                    <div
                        key={`comet-${index}`}
                        className="absolute"
                        style={{
                            top: comet.top,
                            left: comet.left,
                            animation: `cometMove ${comet.speed}s ${comet.delay}s infinite linear`,
                            zIndex: 1
                        }}
                    >
                        <div className="relative">
                            <div
                                className="absolute rounded-full bg-blue-100"
                                style={{
                                    width: `${comet.size}px`,
                                    height: `${comet.size}px`,
                                    boxShadow: `0 0 ${comet.size * 2}px rgba(191, 219, 254, 0.8)`
                                }}
                            ></div>
                            <div
                                className="absolute rounded-full bg-gradient-to-r from-blue-100 via-blue-100/50 to-transparent"
                                style={{
                                    top: `${comet.size / 2}px`,
                                    left: `-${comet.tailLength}px`,
                                    transform: 'translateY(-50%)',
                                    height: `${comet.size}px`,
                                    width: `${comet.tailLength}px`
                                }}
                            ></div>
                        </div>
                    </div>
                ))}

                {/* Nebula-like effects */}
                <div className="absolute top-1/4 left-1/4 w-1/2 h-1/2 rounded-full bg-purple-900/5 blur-3xl transform rotate-45"></div>
                <div className="absolute top-1/3 right-1/4 w-1/3 h-1/3 rounded-full bg-blue-900/5 blur-3xl"></div>
                <div className="absolute bottom-1/4 right-1/3 w-1/3 h-1/4 rounded-full bg-cyan-900/5 blur-3xl"></div>

                {/* Distant galaxy */}
                <div className="absolute top-1/6 right-1/6 w-32 h-6 bg-gradient-to-r from-purple-500/10 via-pink-500/10 to-purple-500/10 rounded-full blur-md transform rotate-45"></div>

                {/* Sun glow from corner */}
                <div className="absolute -bottom-16 -left-16 w-64 h-64 rounded-full bg-gradient-to-r from-yellow-500/30 via-orange-500/20 to-transparent blur-xl"></div>
            </div>

            {/* Slides */}
            <div className="relative h-full">
                {carouselSlides.map((slide, index) => {
                    const theme = getPlanetTheme(index);

                    return (
                        <div
                            key={index}
                            className={`absolute inset-0 flex items-center justify-center transition-opacity duration-500 ${index === currentIndex ? "opacity-100" : "opacity-0"}`}
                        >
                            {/* Slide content container */}
                            <div className="relative w-full h-full">
                                {/* Dynamic Background Effect for each slide */}
                                <div className="absolute inset-0 overflow-hidden">
                                    <div className={`absolute inset-0 bg-gradient-to-b ${theme.primary}`}>
                                        {/* Orbital rings */}
                                        {[...Array(4)].map((_, i) => (
                                            <div
                                                key={i}
                                                className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border"
                                                style={{
                                                    width: `${(i + 1) * 20}%`,
                                                    height: `${(i + 1) * 20}%`,
                                                    borderColor: `rgba(255, 255, 255, ${0.1 - i * 0.02})`,
                                                    transform: `translate(-50%, -50%) rotate(${i * 15}deg)`,
                                                }}
                                            ></div>
                                        ))}

                                        {/* Central planet for this slide */}
                                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                                            <div
                                                className={`rounded-full bg-${theme.accent}-700`}
                                                style={{
                                                    width: '180px',
                                                    height: '180px',
                                                    boxShadow: `0 0 60px rgba(255, 255, 255, 0.2), inset -40px -40px 80px rgba(0, 0, 0, 0.6)`,
                                                    animation: 'planetRotate 30s linear infinite'
                                                }}
                                            >
                                                {/* Planet surface texture */}
                                                {[...Array(8)].map((_, i) => (
                                                    <div
                                                        key={i}
                                                        className="absolute rounded-full bg-gray-900/20"
                                                        style={{
                                                            width: `${10 + Math.random() * 30}px`,
                                                            height: `${10 + Math.random() * 20}px`,
                                                            top: `${Math.random() * 100}%`,
                                                            left: `${Math.random() * 100}%`,
                                                        }}
                                                    ></div>
                                                ))}
                                            </div>

                                            {/* Moon or satellite */}
                                            {index % 2 === 0 && (
                                                <div
                                                    className="absolute rounded-full bg-gray-300"
                                                    style={{
                                                        width: '30px',
                                                        height: '30px',
                                                        animation: 'moonOrbit 10s linear infinite',
                                                        boxShadow: 'inset -10px -10px 20px rgba(0, 0, 0, 0.4)',
                                                    }}
                                                ></div>
                                            )}
                                        </div>
                                    </div>

                                    {/* Gradient overlay */}
                                    <div className="absolute inset-0 bg-gradient-to-t from-gray-950 via-gray-950/70 to-gray-950/20"></div>
                                </div>

                                {/* Content overlay */}
                                <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-8">
                                    {/* Cosmic symbol */}
                                    <div className="mb-4">
                                        <svg viewBox="0 0 100 40" className="w-16 h-8 mx-auto">
                                            <path d="M50,5 L90,20 L50,35 L10,20 Z" fill="none" stroke="rgba(217, 119, 6, 0.5)" strokeWidth="1" />
                                            <circle cx="50" cy="20" r="5" fill="none" stroke="rgba(217, 119, 6, 0.5)" strokeWidth="1" />
                                            <path d="M42,20 L58,20" stroke="rgba(217, 119, 6, 0.5)" strokeWidth="1" />
                                            <path d="M50,15 L50,25" stroke="rgba(217, 119, 6, 0.5)" strokeWidth="1" />
                                        </svg>
                                    </div>

                                    {/* Title with decorative elements */}
                                    <h3 className="text-3xl md:text-4xl font-bold mb-2 relative inline-block">
                                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-amber-400 to-amber-300">{slide.title}</span>
                                        <div className="absolute -bottom-1 left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-amber-700/40 to-transparent"></div>
                                    </h3>

                                    {/* Subtitle */}
                                    <p className="text-lg md:text-xl text-teal-500 mb-4">{slide.subtitle}</p>

                                    {/* Description */}
                                    <p className="text-amber-100/80 max-w-xl">{slide.description}</p>

                                    {/* CTA Button */}
                                    <button className="mt-6 px-6 py-2 bg-gradient-to-r from-amber-700 to-amber-600 hover:from-amber-600 hover:to-amber-500 text-amber-100 rounded-full transition-all duration-300 transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:ring-opacity-50 shadow-lg">
                                        Explore {slide.title}
                                    </button>

                                    {/* Decorative element */}
                                    <div className="mt-6">
                                        <svg viewBox="0 0 100 30" className="w-24 h-6 mx-auto opacity-60">
                                            <path d="M10,15 C20,5 35,10 50,15 C65,10 80,5 90,15" stroke="rgba(217, 119, 6, 0.3)" fill="none" strokeWidth="1" />
                                            <path d="M10,15 C20,25 35,20 50,15 C65,20 80,25 90,15" stroke="rgba(217, 119, 6, 0.3)" fill="none" strokeWidth="1" />
                                            <circle cx="50" cy="15" r="3" fill="rgba(217, 119, 6, 0.3)" />
                                        </svg>
                                    </div>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Navigation arrows */}
            <button
                onClick={prevSlide}
                className="absolute left-4 top-1/2 transform -translate-y-1/2 p-2 bg-gray-900/50 text-amber-400 hover:text-amber-300 hover:bg-gray-900/70 rounded-full transition-all duration-300 focus:outline-none border border-amber-700/30"
                aria-label="Previous slide"
            >
                <ChevronLeft size={24} />
            </button>

            <button
                onClick={nextSlide}
                className="absolute right-4 top-1/2 transform -translate-y-1/2 p-2 bg-gray-900/50 text-amber-400 hover:text-amber-300 hover:bg-gray-900/70 rounded-full transition-all duration-300 focus:outline-none border border-amber-700/30"
                aria-label="Next slide"
            >
                <ChevronRight size={24} />
            </button>

            {/* Navigation dots */}
            <div className="absolute bottom-6 left-0 right-0 flex justify-center gap-2">
                {carouselSlides.map((_, index) => (
                    <button
                        key={index}
                        onClick={() => goToSlide(index)}
                        className={`w-3 h-3 rounded-full transition-all duration-300 focus:outline-none border border-amber-700/50 ${index === currentIndex ? "bg-amber-500 w-6" : "bg-gray-800 hover:bg-amber-700/50"}`}
                        aria-label={`Go to slide ${index + 1}`}
                    />
                ))}
            </div>

            {/* Animation keyframes */}
            <style jsx>{`
                @keyframes cometMove {
                    from { transform: translateX(0); }
                    to { transform: translateX(150vw); }
                }
                @keyframes planetRotate {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                @keyframes moonOrbit {
                    0% { transform: rotate(0deg) translateX(130px); }
                    100% { transform: rotate(360deg) translateX(130px); }
                }
                @keyframes orbit0 {
                    0% { transform: translateY(-50px); }
                    50% { transform: translateY(50px); }
                    100% { transform: translateY(-50px); }
                }
                @keyframes orbit1 {
                    0% { transform: translateX(-30px); }
                    50% { transform: translateX(30px); }
                    100% { transform: translateX(-30px); }
                }
                @keyframes orbit2 {
                    0% { transform: translate(-40px, -40px); }
                    50% { transform: translate(40px, 40px); }
                    100% { transform: translate(-40px, -40px); }
                }
                @keyframes orbit3 {
                    0% { transform: translate(20px, -30px); }
                    50% { transform: translate(-20px, 30px); }
                    100% { transform: translate(20px, -30px); }
                }
                @keyframes orbit4 {
                    0% { transform: translate(-10px, 40px); }
                    50% { transform: translate(10px, -40px); }
                    100% { transform: translate(-10px, 40px); }
                }
                @keyframes orbit5 {
                    0% { transform: translateY(-20px) rotate(0deg); }
                    50% { transform: translateY(20px) rotate(180deg); }
                    100% { transform: translateY(-20px) rotate(360deg); }
                }
                @keyframes orbit6 {
                    0% { transform: translateX(-35px) rotate(0deg); }
                    50% { transform: translateX(35px) rotate(180deg); }
                    100% { transform: translateX(-35px) rotate(360deg); }
                }
                @keyframes orbit7 {
                    0% { transform: translate(-25px, -25px) rotate(0deg); }
                    50% { transform: translate(25px, 25px) rotate(180deg); }
                    100% { transform: translate(-25px, -25px) rotate(360deg); }
                }
            `}</style>
        </div>
    );
};

export default CarousalOfHome;