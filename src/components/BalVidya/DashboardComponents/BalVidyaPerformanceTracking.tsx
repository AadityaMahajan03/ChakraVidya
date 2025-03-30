import React from 'react';
import { Trophy, Star, Rocket, Brain, Award, Home, Book, Medal, FileText } from 'lucide-react';

const BalVidyaPerformanceTracking = () => {
  const subjects = [
    { subject: "Math", percentage: 80, color: "bg-blue-500", icon: <Brain className="text-blue-500" size={20} />, emoji: "🧮" },
    { subject: "Science", percentage: 75, color: "bg-green-500", icon: <Rocket className="text-green-500" size={20} />, emoji: "🔬" },
    { subject: "English", percentage: 92, color: "bg-purple-500", icon: <Award className="text-purple-500" size={20} />, emoji: "📚" },
    { subject: "Social Studies", percentage: 67, color: "bg-yellow-500", icon: <Star className="text-yellow-500" size={20} />, emoji: "🌎" },
  ];

  return (
    <div className="bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-400 min-h-screen w-full p-4 md:p-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6 w-full">
        <h1 className="text-3xl font-bold text-white flex items-center mb-4 md:mb-0">
          <Trophy className="text-yellow-300 mr-3" size={36} />
          BalVidya Learning Dashboard
        </h1>
        <div className="flex space-x-4">
          <div className="bg-yellow-300/20 backdrop-blur-sm px-4 py-2 rounded-full border border-yellow-200/50 flex items-center">
            <Star className="text-yellow-300 mr-2" size={18} />
            <span className="font-bold text-white text-lg">314</span>
            <span className="ml-1 text-white/80">Stars</span>
          </div>
          <div className="bg-blue-300/20 backdrop-blur-sm px-4 py-2 rounded-full border border-blue-200/50 flex items-center">
            <Medal className="text-blue-300 mr-2" size={18} />
            <span className="font-bold text-white text-lg">Level 7</span>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="bg-white/10 backdrop-blur-md rounded-xl mb-6 p-2 border border-white/30 overflow-x-auto">
        <div className="flex space-x-2">
          <button className="bg-white/20 hover:bg-white/30 text-white px-4 py-2 rounded-lg flex items-center transition-all">
            <Home size={18} className="mr-2" />
            <span>Home</span>
          </button>
          <button className="bg-white/30 text-white px-4 py-2 rounded-lg flex items-center transition-all">
            <Trophy size={18} className="mr-2" />
            <span>Progress</span>
          </button>
          <button className="bg-white/20 hover:bg-white/30 text-white px-4 py-2 rounded-lg flex items-center transition-all">
            <Book size={18} className="mr-2" />
            <span>Lessons</span>
          </button>
          <button className="bg-white/20 hover:bg-white/30 text-white px-4 py-2 rounded-lg flex items-center transition-all">
            <FileText size={18} className="mr-2" />
            <span>Assignments</span>
          </button>
        </div>
      </div>

      {/* Main Content - Full Width Container */}
      <div className="w-full grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Learning Map - Expanded */}
        <div className="lg:col-span-2 bg-white/10 backdrop-blur-md rounded-xl overflow-hidden shadow-lg border border-white/30 transform transition-all duration-300 hover:shadow-xl relative">
          {/* Glow orbs */}
          <div className="absolute -top-20 -left-20 w-40 h-40 bg-blue-400 rounded-full filter blur-3xl opacity-30"></div>
          <div className="absolute -bottom-20 -right-20 w-40 h-40 bg-pink-400 rounded-full filter blur-3xl opacity-30"></div>

          <div className="bg-gradient-to-r from-blue-600/80 to-indigo-600/80 backdrop-blur-sm py-3 px-4 border-b border-white/20">
            <h3 className="font-bold text-white text-lg">Your Learning Journey</h3>
          </div>
          <div className="p-5 relative z-10">
            <div className="relative w-full h-64 bg-indigo-900/20 backdrop-blur-sm rounded-lg flex items-center justify-center mb-6 border border-white/20 overflow-hidden">
              {/* Animated stars in background */}
              <div className="absolute inset-0 overflow-hidden">
                {[...Array(30)].map((_, i) => (
                  <div
                    key={i}
                    className="absolute w-1 h-1 bg-white rounded-full animate-pulse"
                    style={{
                      top: `${Math.random() * 100}%`,
                      left: `${Math.random() * 100}%`,
                      animationDelay: `${Math.random() * 2}s`,
                      opacity: Math.random() * 0.7 + 0.3
                    }}
                  />
                ))}
              </div>

              {/* Journey path with visual elements */}
              <div className="absolute bottom-0 w-full h-32 bg-gradient-to-t from-green-500/30 to-transparent"></div>

              {/* Character */}
              <div className="absolute bottom-6 left-1/4 text-4xl animate-bounce" style={{ animationDuration: '2s' }}>🚶</div>

              {/* Trophy */}
              <div className="absolute top-10 right-1/6 text-4xl animate-pulse" style={{ animationDuration: '3s' }}>🏆</div>

              {/* Path with milestones */}
              <div className="absolute w-3/4 border-t-4 border-dashed border-yellow-400/70 bottom-16 left-1/8">
                <div className="flex items-center justify-between relative">
                  {[0, 1, 2, 3, 4].map((i) => (
                    <div
                      key={i}
                      className={`absolute w-12 h-12 rounded-full flex items-center justify-center text-lg
                      ${i <= 2 ? 'bg-yellow-400 border-2 border-white animate-pulse' : 'bg-white/20'}`}
                      style={{
                        animationDelay: `${i * 0.2}s`,
                        left: `${i * 25}%`,
                        top: '-24px'
                      }}
                    >
                      {i <= 2 ? '✓' : i + 1}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 w-full">
              <div className="bg-blue-600/20 backdrop-blur-sm p-4 rounded-lg text-center border border-white/20 hover:bg-blue-600/30 transition-all duration-300">
                <div className="text-2xl animate-pulse">🔥</div>
                <p className="font-medium mt-2 text-white">5 Day Streak!</p>
              </div>
              <div className="bg-purple-600/20 backdrop-blur-sm p-4 rounded-lg text-center border border-white/20 hover:bg-purple-600/30 transition-all duration-300">
                <div className="text-2xl animate-pulse">⭐</div>
                <p className="font-medium mt-2 text-white">21 Stars This Week</p>
              </div>
              <div className="bg-green-600/20 backdrop-blur-sm p-4 rounded-lg text-center border border-white/20 hover:bg-green-600/30 transition-all duration-300">
                <div className="text-2xl animate-pulse">📚</div>
                <p className="font-medium mt-2 text-white">4 Lessons Complete</p>
              </div>
              <div className="bg-yellow-600/20 backdrop-blur-sm p-4 rounded-lg text-center border border-white/20 hover:bg-yellow-600/30 transition-all duration-300">
                <div className="text-2xl animate-pulse">🎯</div>
                <p className="font-medium mt-2 text-white">Next Goal: Science Quiz</p>
              </div>
            </div>

            <div className="mt-6 flex justify-center">
              <button className="px-6 py-3 bg-gradient-to-r from-blue-500 to-indigo-500 text-white font-bold rounded-lg shadow-md hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1 border border-white/30 relative overflow-hidden group">
                <span className="relative z-10 flex items-center">
                  <Trophy size={18} className="mr-2" />
                  View All Achievements
                </span>
                <span className="absolute inset-0 bg-white/20 transform scale-x-0 group-hover:scale-x-100 transition-transform origin-left duration-500"></span>
              </button>
            </div>
          </div>
        </div>

        {/* Skills Card - Now taller to match */}
        <div className="bg-white/10 backdrop-blur-md rounded-xl overflow-hidden shadow-lg border border-white/30 transform transition-all duration-300 hover:shadow-xl relative">
          <div className="absolute -top-20 -right-20 w-40 h-40 bg-purple-400 rounded-full filter blur-3xl opacity-30"></div>

          <div className="bg-gradient-to-r from-purple-600/80 to-pink-600/80 backdrop-blur-sm py-3 px-4 border-b border-white/20">
            <h3 className="font-bold text-white text-lg">Your Super Skills</h3>
          </div>
          <div className="p-5 relative z-10">
            <ul className="space-y-6">
              {subjects.map(({ subject, percentage, color, icon, emoji }) => (
                <li key={subject} className="transform transition-all duration-300 hover:translate-x-1">
                  <div className="flex items-center mb-2">
                    <span className="mr-2 text-2xl">{emoji}</span>
                    <span className="font-medium text-white text-lg">{subject}</span>
                    <div className="ml-auto flex items-center">
                      <span className="font-bold text-white text-lg">{percentage}%</span>
                      {percentage >= 90 && <span className="ml-1 animate-pulse">🌟</span>}
                    </div>
                  </div>
                  <div className="relative w-full">
                    <div className="w-full h-6 bg-black/20 backdrop-blur-sm rounded-full overflow-hidden border border-white/20">
                      <div
                        className={`${color} h-full rounded-full transition-all duration-1000 ease-out`}
                        style={{ width: `${percentage}%` }}
                      ></div>
                    </div>

                    {/* Character positioning based on progress */}
                    <div
                      className="absolute top-0 transform -translate-y-1/2 animate-bounce"
                      style={{
                        left: `${Math.min(Math.max(percentage - 5, 0), 95)}%`,
                        animationDuration: '2s'
                      }}
                    >
                      {percentage < 50 ? '🐢' : percentage < 75 ? '🐰' : '🚀'}
                    </div>
                  </div>
                </li>
              ))}
            </ul>

            <div className="mt-8 space-y-4">
              <button className="w-full py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-bold rounded-lg shadow-md hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1 border border-white/30 relative overflow-hidden group text-lg">
                <span className="relative z-10">Collect Today's Rewards! 🎁</span>
                <span className="absolute inset-0 bg-white/20 transform scale-x-0 group-hover:scale-x-100 transition-transform origin-left duration-500"></span>
              </button>

              <button className="w-full py-3 bg-indigo-500/30 text-white font-bold rounded-lg shadow-md hover:shadow-lg transition-all duration-300 border border-white/30 hover:bg-indigo-500/50 text-lg">
                Practice Skills 🎮
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Additional Section - Recent Activity */}
      <div className="mt-6 bg-white/10 backdrop-blur-md rounded-xl overflow-hidden shadow-lg border border-white/30 w-full p-5 relative">
        <div className="absolute -bottom-20 -left-20 w-40 h-40 bg-green-400 rounded-full filter blur-3xl opacity-20"></div>

        <h3 className="font-bold text-white text-lg mb-4">📝 Recent Activity</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="bg-white/10 backdrop-blur-sm p-4 rounded-lg border border-white/20 hover:bg-white/20 transition-all">
            <div className="flex items-start mb-2">
              <div className="bg-blue-500/30 p-2 rounded-lg mr-3">📚</div>
              <div>
                <h4 className="font-medium text-white">Math Quiz Completed</h4>
                <p className="text-white/70 text-sm">Score: 85% • Today</p>
              </div>
            </div>
            <div className="mt-2 text-white/80 text-sm">Great job on fractions! Try again to reach 90%</div>
          </div>

          <div className="bg-white/10 backdrop-blur-sm p-4 rounded-lg border border-white/20 hover:bg-white/20 transition-all">
            <div className="flex items-start mb-2">
              <div className="bg-purple-500/30 p-2 rounded-lg mr-3">🏆</div>
              <div>
                <h4 className="font-medium text-white">Achievement Unlocked</h4>
                <p className="text-white/70 text-sm">Science Explorer • Yesterday</p>
              </div>
            </div>
            <div className="mt-2 text-white/80 text-sm">Completed all lessons in the Solar System chapter</div>
          </div>

          <div className="bg-white/10 backdrop-blur-sm p-4 rounded-lg border border-white/20 hover:bg-white/20 transition-all">
            <div className="flex items-start mb-2">
              <div className="bg-green-500/30 p-2 rounded-lg mr-3">🔥</div>
              <div>
                <h4 className="font-medium text-white">5-Day Streak!</h4>
                <p className="text-white/70 text-sm">Keep it up! • Today</p>
              </div>
            </div>
            <div className="mt-2 text-white/80 text-sm">You're on fire! 10-day streak unlocks special rewards</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BalVidyaPerformanceTracking;