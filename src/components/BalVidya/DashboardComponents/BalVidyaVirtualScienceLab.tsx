import { Beaker, Microscope, Atom, Zap, Rocket, AlertTriangle } from "lucide-react";
import React from "react";

const BalVidyaVirtualScienceLab = () => {
    return (
        <div className="w-full h-screen bg-gradient-to-br from-blue-700 via-indigo-600 to-purple-700 text-white p-6 overflow-auto">
            {/* Decorative Elements */}
            <div className="absolute top-20 right-40 w-32 h-32 bg-green-400 rounded-full opacity-30 blur-md"></div>
            <div className="absolute bottom-40 left-20 w-40 h-40 bg-yellow-500 rounded-full opacity-20 blur-md"></div>
            <div className="absolute top-1/3 left-1/4 w-24 h-24 bg-cyan-400 rounded-full opacity-30 blur-md"></div>

            <div className="container mx-auto relative">
                <div className="flex items-center justify-between mb-8 pt-4">
                    <h2 className="text-3xl md:text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-300 to-green-300">COSMIC SCIENCE LABORATORY</h2>
                    <Rocket className="text-yellow-300 h-10 w-10" />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    {labSections.map((section, index) => (
                        <div
                            key={index}
                            className="border-2 border-opacity-70 rounded-xl overflow-hidden hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2 bg-indigo-900 bg-opacity-40 backdrop-blur-sm"
                            style={{ borderColor: section.borderColor }}
                        >
                            <div className={`h-48 flex items-center justify-center bg-gradient-to-r ${section.gradientFrom} ${section.gradientTo}`}>
                                <div className="relative">
                                    <div className="absolute -inset-6 bg-white opacity-20 rounded-full animate-pulse"></div>
                                    <section.icon size={64} className="text-white" />
                                </div>
                            </div>
                            <div className="p-5">
                                <h3 className="font-bold text-xl mb-3" style={{ color: section.textColor }}>{section.title}</h3>
                                <p className="text-gray-200 mb-4">{section.description}</p>
                                <button
                                    className={`w-full py-3 rounded-lg font-bold text-white flex items-center justify-center space-x-2 transition-all duration-300 hover:opacity-90 hover:scale-105 bg-gradient-to-r ${section.buttonGradientFrom} ${section.buttonGradientTo}`}
                                >
                                    <Zap size={18} />
                                    <span>BEGIN EXPERIMENT</span>
                                </button>
                            </div>
                        </div>
                    ))}
                </div>

                <div className="mt-6 p-6 bg-yellow-900 bg-opacity-30 rounded-xl border-2 border-yellow-400 shadow-lg backdrop-blur-sm">
                    <div className="flex items-start">
                        <div className="mr-4 mt-1">
                            <div className="bg-yellow-500 p-3 rounded-lg">
                                <AlertTriangle size={24} className="text-yellow-900" />
                            </div>
                        </div>
                        <div>
                            <h3 className="font-bold text-xl mb-2 text-yellow-300">SAFETY PROTOCOLS ACTIVE!</h3>
                            <p className="text-yellow-100">All experiments are simulated in a safe environment, but remember to apply these safety principles in real-life science classes too! Your mission control ensures secure exploration of all scientific frontiers.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

const labSections = [
    {
        title: "CHEMICAL REACTION CHAMBER",
        description: "Mix compounds and create dazzling reactions in our advanced virtual laboratory. Safe for all cosmic explorers!",
        icon: Zap,
        gradientFrom: "from-blue-600",
        gradientTo: "to-blue-400",
        borderColor: "#60a5fa",
        textColor: "#93c5fd",
        buttonGradientFrom: "from-blue-600",
        buttonGradientTo: "to-cyan-600"
    },
    {
        title: "BIOLOGICAL DISCOVERY ZONE",
        description: "Journey inside cells, explore DNA, and witness the wonders of life forms across the universe.",
        icon: Microscope,
        gradientFrom: "from-green-600",
        gradientTo: "to-green-400",
        borderColor: "#4ade80",
        textColor: "#86efac",
        buttonGradientFrom: "from-green-600",
        buttonGradientTo: "to-teal-600"
    },
    {
        title: "QUANTUM PHYSICS ENGINE",
        description: "Control gravity, manipulate time, and explore the fundamental forces that shape our universe!",
        icon: Atom,
        gradientFrom: "from-purple-600",
        gradientTo: "to-purple-400",
        borderColor: "#c084fc",
        textColor: "#d8b4fe",
        buttonGradientFrom: "from-purple-600",
        buttonGradientTo: "to-pink-600"
    }
];

export default BalVidyaVirtualScienceLab;