import { useState, useEffect } from "react";
import {
    Home,
    BookOpen,
    Play,
    BarChart2,
    Users,
    BookMarked,
    Shield,
    Bot,
    Beaker,
    Glasses,
    Star,
    Calendar,
    Award,
    Microscope,
    Atom
} from "lucide-react";
import BalVidyaDashSideNav from "./BalVidyaDashSideNav";
import BalVidyaHomeStudentDashboard from "./DashboardComponents/BalVidyaHomeStudentDashboard";
import BalVidyaLearningModules from "./DashboardComponents/BalVidyaLearningModules";
import BalVidyaFunActivities from "./DashboardComponents/BalVidyaFunActivities";
import BalVidyaPerformanceTracking from "./DashboardComponents/BalVidyaPerformanceTracking";
import BalVidyaHomeworkClassroom from "./DashboardComponents/BalVidyaHomeworkClassroom";
import BalVidyaCommunitySocial from "./DashboardComponents/BalVidyaCommunitySocial";
import BalVidyaSafetyParentalControl from "./DashboardComponents/BalVidyaSafetyParentalControl";
import BalVidyaAILearningAssistant from "./DashboardComponents/BalVidyaAILearningAssistant";
import BalVidyaVirtualScienceLab from "./DashboardComponents/BalVidyaVirtualScienceLab";
import BalVidyaAugmentedRealityLearning from "./DashboardComponents/BalVidyaAugmentedRealityLearning";

export default function BalVidyaDashboard({ email }) {
    const [activeSection, setActiveSection] = useState("home");
    const [navItems, setNav] = useState([]);

    // useEffect(() => {
    //     setNav([
    //         { id: "home", label: "Home", icon: <Home size={18} /> },
    //         { id: "learning", label: "Learning", icon: <BookOpen size={18} /> },
    //         { id: "activities", label: "Activities", icon: <Play size={18} /> },
    //         { id: "performance", label: "Performance", icon: <BarChart2 size={18} /> },
    //         { id: "community", label: "Community", icon: <Users size={18} /> },
    //         { id: "homework", label: "Homework", icon: <BookMarked size={18} /> },
    //         { id: "safety", label: "Safety", icon: <Shield size={18} /> },
    //         { id: "aiTutor", label: "AI Tutor", icon: <Bot size={18} /> },
    //         { id: "virtualLab", label: "Virtual Lab", icon: <Microscope size={18} /> },
    //         { id: "arLearning", label: "AR Learning", icon: <Glasses size={18} /> }
    //     ]);
    // }, []); // Runs only once when the component mounts


    const navForKindergarten = [
        { id: "home", label: "Home", icon: <Home size={18} /> }
    ];

    const navForGrade1AndAbove = [
        { id: "home", label: "Home", icon: <Home size={18} /> },
        { id: "learning", label: "Learning", icon: <BookOpen size={18} /> },
        { id: "aiTutor", label: "AI Tutor", icon: <Bot size={18} /> },
        { id: "arLearning", label: "AR Learning", icon: <Glasses size={18} /> },
        { id: "performance", label: "Performance", icon: <BarChart2 size={18} /> }
    ];

    const navForGrade2AndAbove = [
        { id: "home", label: "Home", icon: <Home size={18} /> },
        { id: "learning", label: "Learning", icon: <BookOpen size={18} /> },
        { id: "aiTutor", label: "AI Tutor", icon: <Bot size={18} /> },
        { id: "arLearning", label: "AR Learning", icon: <Glasses size={18} /> },
        { id: "performance", label: "Performance", icon: <BarChart2 size={18} /> }
    ];

    const navForGrade3AndAbove = [
        { id: "home", label: "Home", icon: <Home size={18} /> },
        { id: "learning", label: "Learning", icon: <BookOpen size={18} /> },
        { id: "performance", label: "Performance", icon: <BarChart2 size={18} /> }
    ];

    const [user, setUser] = useState(null);

    //get function that would get the object of user
    const userObject = async () => {
        try {
            const response = await fetch("http://localhost:5000/get-user-object", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ email: email }),
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


    // Fetch user on mount
    useEffect(() => {
        userObject();
    }, []);

    useEffect(() => {
        if (user && user.grade === "Kindergarten") {
            setNav(navForKindergarten);
        }
        else if (user && user.grade === "Grade 1") {
            setNav(navForGrade1AndAbove);
        }
        else if (user && user.grade === "Grade 2") {
            setNav(navForGrade2AndAbove);
        }
        else if (user && user.grade === "Grade 3") {
            setNav(navForGrade3AndAbove);
        }
    }, [user]);



    return (
        <div className="flex h-screen bg-gray-50">
            {/* Sidebar */}
            <BalVidyaDashSideNav navItems={navItems} activeSection={activeSection} setActiveSection={setActiveSection} emailOfUser={email} />


            {/* Main Content */}
            <div className="flex-1 overflow-auto">
                {/* <header className="bg-white shadow-sm p-4 border-b">
                    <h2 className="text-xl font-semibold text-gray-800">
                        {navItems.find(item => item.id === activeSection)?.label}
                    </h2>
                </header> */}

                <main>
                    {activeSection === "home" && (
                        <BalVidyaHomeStudentDashboard emailOfUser={email} />
                    )}

                    {activeSection === "learning" && (
                        <BalVidyaLearningModules />
                    )}

                    {activeSection === "activities" && (
                        <BalVidyaFunActivities />
                    )}

                    {activeSection === "performance" && (
                        <BalVidyaPerformanceTracking />
                    )}

                    {activeSection === "community" && (
                        <BalVidyaCommunitySocial />
                    )}

                    {activeSection === "homework" && (
                        <BalVidyaHomeworkClassroom />
                    )}

                    {activeSection === "safety" && (
                        <BalVidyaSafetyParentalControl />
                    )}

                    {activeSection === "aiTutor" && (
                        <BalVidyaAILearningAssistant />
                    )}

                    {activeSection === "virtualLab" && (
                        <BalVidyaVirtualScienceLab />
                    )}

                    {activeSection === "arLearning" && (
                        <BalVidyaAugmentedRealityLearning />
                    )}
                </main>
            </div>
        </div>
    );
}