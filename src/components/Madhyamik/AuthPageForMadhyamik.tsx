import { useState } from "react";

export default function AuthPageForMadhyamik() {
    const [isRegistering, setIsRegistering] = useState(false);
    const [userType, setUserType] = useState("student");
    const [grade, setGrade] = useState("6");

    return (
        <div>
            <h2>{isRegistering ? "Madhyamik Registration" : "Madhyamik Login"}</h2>
            <form>
                {isRegistering && (
                    <>
                        <label>Name:</label>
                        <input type="text" placeholder="Enter your name" required />
                    </>
                )}

                <label>Email:</label>
                <input type="email" placeholder="Enter your email" required />

                <label>Password:</label>
                <input type="password" placeholder="Enter your password" required />

                {isRegistering && (
                    <>
                        <label>User Type:</label>
                        <select value={userType} onChange={(e) => setUserType(e.target.value)}>
                            <option value="student">Student</option>
                            <option value="teacher">Teacher</option>
                            <option value="parent">Parent</option>
                        </select>

                        {userType === "student" && (
                            <>
                                <label>Grade Level:</label>
                                <select value={grade} onChange={(e) => setGrade(e.target.value)}>
                                    {[...Array(5)].map((_, i) => (
                                        <option key={i} value={i + 6}>{i + 6}</option>
                                    ))}
                                </select>
                            </>
                        )}
                    </>
                )}

                <button type="submit">{isRegistering ? "Register" : "Login"}</button>
            </form>

            <p>
                {isRegistering ? "Already have an account?" : "Don't have an account?"}
                <button onClick={() => setIsRegistering(!isRegistering)}>
                    {isRegistering ? "Login" : "Register"}
                </button>
            </p>
        </div>
    );
}