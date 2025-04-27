import { BrowserRouter } from "react-router-dom";
import { Routes, Route } from "react-router";
import Home from "./routes/Home";
import LoginPage from "./routes/Login";
import SignupPage from "./routes/Signup";
import ContactUs from "./routes/ContactUs";
import ResumeBuilder from "./routes/ResumeBuilder";
import ResumeUpload from "./routes/ResumeUpload"; // Import the Jobseekers component
import AboutUs from "./routes/AboutUs";
import Jobseekers from "./routes/Dashboard";
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { Dashboard } from "@mui/icons-material";
import InstructionalVideosPage from "./routes/InstructionalVideosPage";
import ResumeViewer from "./components/ResumeViewer";
import JobseekerDashboard from "./routes/Jobseeker-dashboard";
import ForgotPasswordPage from "./routes/forgotPassword";
import PostJob from "./components/PostJob";
import EmployerDashboard from "./routes/Employer-dashboard";
import JobListings from "./components/JobListings";
import AIMLtest from "./components/tests/AIML";
import JobDetails from "./routes/JobDetails";
import ResultsPage from "./components/ResultsPage";
import CreateProfile from "./routes/GetStarted";
import SuccessStories from "./routes/SuccessStories";
import ManageJobs from "./components/ManageJobs";
import AIMockInterview from "./components/AIMockInterview";
import MeetingRoom from "./components/MeetingRoom";

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#ffffff',
    },
  },
  typography: {
    fontFamily: 'Roboto, sans-serif',
  },
});

function App() {

  return (
    <div className="App">
      <ThemeProvider theme={theme}>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />
            <Route path="/contactus" element={<ContactUs />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/jobseeker-dashboard" element={<JobseekerDashboard />} />
            <Route path="/employer-dashboard" element={<EmployerDashboard />} />
            <Route path="/instructional-videos" element={<InstructionalVideosPage />} />
            <Route path="/success-stories" element={<SuccessStories />} />
            <Route path="/resume-builder" element={<ResumeBuilder />} />
            <Route path="/resume-viewer" element={<ResumeViewer />} />
            <Route path="/resume-upload" element={<ResumeUpload />} />
            <Route path="/jobseekers" element={<Jobseekers />} /> 
            <Route path="/aboutus" element={<AboutUs />} /> 
            <Route path="/videos" element={<InstructionalVideosPage/>} />
            <Route path='/forgot-password' element={<ForgotPasswordPage/>} />
            <Route path="/post-job" element={<PostJob />} />
            <Route path="/job-listings" element={<JobListings />} />
            <Route path="/job/:jobRole" element={<JobDetails />} />
            <Route path="/competency-test/:jobId" element={<AIMLtest />} />
            <Route path="/test-results/:testId/:jobId" element={<ResultsPage />} />
            <Route path="/get-started" element={<CreateProfile />} />
            <Route path="/interview" element={<AIMockInterview />} />
            <Route path="manage-jobs" element={<ManageJobs />} />
            <Route path="/meeting-room/:meetingId" element={<MeetingRoom />} />
          </Routes>
        </BrowserRouter>
      </ThemeProvider>
    </div>
  );
}

export default App;
