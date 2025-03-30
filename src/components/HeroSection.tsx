import React, { useState } from "react";
import Section from "./HeroSection/Section";
import ListSection from "./HeroSection/ListSection";
import GuruSection from "./HeroSection/GuruSection";
import ResourceSection from "./HeroSection/ResourceSection";
import CarousalOfHome from "./HeroSection/CarousalOfHome";
// Krishna Wisdom EdTech Hero Component
const KrishnaWisdomEdTechHero = () => {
    const [categories] = useState([
        "Balvidya (Primary Education) - Foundation learning for young children.",
        "Madhyamik (Secondary Education) - Strengthening core subjects and analytical skills.",
        "Uccha Shiksha (Higher Education) - Advanced studies in specialized fields.",
        "Vishesh Vidya (Specialized Knowledge) - Focus on unique disciplines and deep expertise."
    ]);
    const [features] = useState([
        "AI-driven career mentorship",
        "Personalized preparation guidance",
        "Inclusive learning for dyslexic and ADHD learners",
        "Opportunities for blue-collar workers to upskill"
    ]);
    const [gurus] = useState([
        { name: "Krishna", description: "The greatest teacher of all time." },
        { name: "Vidhur", description: "The wisest person of Mahabharata." },
        { name: "Vivekananda", description: "The wisest guru who has made worldwide recognition of India." },
        { name: "Normal Human", description: "The more relatable companion." }
    ]);
    const [resources] = useState([
        { title: "Bhagavad Gita", description: "A sacred scripture containing Krishna's teachings.", link: "https://www.holy-bhagavad-gita.org/" },
        { title: "Vidhur Neeti", description: "Ethical and political wisdom from Vidhur.", link: "https://www.wisdomlib.org/hinduism/book/vidura-niti" },
        { title: "Complete Works of Swami Vivekananda", description: "Teachings and philosophy of Swami Vivekananda.", link: "https://advaitaashrama.org/complete-works-of-swami-vivekananda/" },
        { title: "Ancient Indian Education", description: "A study on India's traditional learning systems.", link: "https://www.exoticindiaart.com/book/details/ancient-indian-education-bve137/" }
    ]);
    const [cards] = useState([
        {
            title: "Ancient Wisdom",
            content: "Discover the secrets of the cosmos and the hidden knowledge of ancient civilizations."
        },
        {
            title: "Mystic Powers",
            content: "Harness the energy of the universe and transcend the limitations of physical existence."
        },
        {
            title: "Divine Revelations",
            content: "Receive insights from beyond the veil and understand the language of celestial beings."
        }
    ]);

    return (
        <div>
            {/* <h1>Krishna Wisdom EdTech</h1> */}
            <CarousalOfHome />
            <Section title="Vision" cards={cards} />
            <ListSection title="Categories" items={categories} />
            <ListSection title="Features" items={features} />
            <GuruSection gurus={gurus} />
            <ResourceSection resources={resources} />
        </div>
    );
};

export default KrishnaWisdomEdTechHero;
