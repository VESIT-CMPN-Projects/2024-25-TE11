import { initializeApp } from 'firebase/app';
import { getFirestore, doc, getDoc, updateDoc, collection, getDocs } from 'firebase/firestore';
import { getAuth } from 'firebase/auth';

// Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyDQ1KzJbF-9938rllR2UQ_LuEdFsdmn7Gs",
    authDomain: "skillbridgeai-bd59d.firebaseapp.com",
    projectId: "skillbridgeai-bd59d",
    storageBucket: "skillbridgeai-bd59d.appspot.com",
    messagingSenderId: "137919663891",
    appId: "1:137919663891:web:2bcd156b6b81d2cc6b54e6",
    measurementId: "G-H2SYDTG83R"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const auth = getAuth(app);

// Get current user's profile
export const getUserProfile = async () => {
  try {
    const user = auth.currentUser;
    if (!user) throw new Error('User not authenticated');

    const userDoc = await getDoc(doc(db, 'userProfiles', user.uid));
    if (userDoc.exists()) {
      return userDoc.data();
    } else {
      throw new Error('User profile not found');
    }
  } catch (error) {
    console.error('Error fetching user profile:', error);
    throw error;
  }
};

// Get job roles and skills data
export const getJobRolesSkills = async () => {
      return {
        "Full Stack Developer": ["HTML/CSS", "Javascript", "Node.js", "Frontend Frameworks", "Backend Development", "Version Control", "Database Management", "Mobile App Development", "APIs", "Testing/Debugging", "System Design and Architecture"],
        "AI/ML Engineer": ["Python", "Machine Learning Frameworks", "Deep Learning", "Data Science Tools", "NLP", "Computer Vision", "Data Visualization", "Big Data Tools", "Model Deployment", "Data Analysis and Statistics"],
        "Cloud and DevOps Engineer": ["Cloud Platforms", "Infrastructure as Code", "Containerization", "Orchestration", "CI/CD", "Monitoring and Logging", "Scripting", "Configuration Management", "Networking", "System Design and Architecture", "Operating Systems"],
        "Software Engineer": ["Version Control", "Database Management", "Networking and Security Protocols", "Blockchain Development", "Cybersecurity", "Security Tools", "Cryptography", "Identity Management", "DevSecOps"]
      };
};

// Get available courses
export const getAvailableCourses = async () => {
  try {
    const coursesSnapshot = await getDocs(collection(db, 'courses'));
    const courses = [];
    coursesSnapshot.forEach((doc) => {
      courses.push({
        id: doc.id,
        ...doc.data()
      });
    });
    return courses;
  } catch (error) {
    console.error('Error fetching courses:', error);
    throw error;
  }
};

// Get available competency tests
export const getCompetencyTests = async () => {
  try {
    const testsSnapshot = await getDocs(collection(db, 'competencyTests'));
    const tests = [];
    testsSnapshot.forEach((doc) => {
      tests.push({
        id: doc.id,
        ...doc.data()
      });
    });
    return tests;
  } catch (error) {
    console.error('Error fetching competency tests:', error);
    throw error;
  }
};