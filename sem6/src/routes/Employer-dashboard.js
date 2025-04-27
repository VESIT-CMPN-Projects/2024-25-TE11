import React, { useState } from "react";
import { Box, Typography } from "@mui/material";
import Sidebar from "../components/Sidebar";
import JobListings from "../components/JobListings";
import PostJob from "../components/PostJob";
import ManageJobs from "../components/ManageJobs";
import EmployerProfile from "../components/EmployerProfile";
import MentorSessions from "../components/MentorSessions";

const EmployerDashboard = () => {
    const [activePage, setActivePage] = useState("profile");

    const renderComponent = () => {
        switch (activePage) {
            case "profile":
                return <EmployerProfile />;
            case "job-listings":
                return <JobListings />;
            case "post-job":
                return <PostJob />;
            case "manage-jobs":
                return <ManageJobs />;
            case "sessions":
                return <MentorSessions />;
            case "dashboard":
            default:
                return <Typography variant="h4">Welcome to Employer Dashboard</Typography>;
        }
    };

    return (
        <Box sx={{ display: "flex", height: "100vh" }}>
            <Sidebar setActivePage={setActivePage} />
            <Box sx={{ flexGrow: 1, p: 4 }}>{renderComponent()}</Box>
        </Box>
    );
};

export default EmployerDashboard;
