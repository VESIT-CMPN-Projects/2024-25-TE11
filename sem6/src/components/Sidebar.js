import React, { useState } from "react";
import {
  Box, Drawer, IconButton, List, ListItem, ListItemButton, ListItemIcon, ListItemText, Tooltip, Typography, Divider
} from "@mui/material";
import { Menu, Group, Work, PostAdd, ManageAccounts, ExitToApp } from "@mui/icons-material";

const Sidebar = ({ setActivePage }) => {
  const [open, setOpen] = useState(true);
  const toggleDrawer = () => setOpen(!open);
  
  const [activeItem, setActiveItem] = useState("profile");
  
  const menuItems = [
    { text: "My Profile", icon: <ManageAccounts />, page: "profile" },
    { text: "Applied candidates", icon: <Group />, page: "job-listings" },
    { text: "Post a Job", icon: <PostAdd />, page: "post-job" },
    { text: "Manage Jobs", icon: <Work />, page: "manage-jobs" },
    { text: "Mentor Sessions", icon: <Group />, page: "sessions" },
    { text: "Log Out", icon: <ExitToApp />, page: "/" }
  ];
  
  const handleMenuClick = (page) => {
    setActiveItem(page);
    setActivePage(page);
  };
  
  return (
    <Drawer
      variant="permanent"
      sx={{
        width: open ? 240 : 70,
        flexShrink: 0,
        "& .MuiDrawer-paper": {
          width: open ? 240 : 70,
          transition: "width 0.3s ease-in-out",
          backgroundColor: "#f8f9fa",
          boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
          borderRight: "none",
          overflowX: "hidden"
        }
      }}
    >
      <Box sx={{ display: "flex", justifyContent: open ? "space-between" : "center", alignItems: "center", p: 2 }}>
        {open && <Typography variant="subtitle1" sx={{ fontWeight: 600, color: "#3a86ff" }}>Dashboard</Typography>}
        <IconButton 
          onClick={toggleDrawer}
          sx={{
            backgroundColor: "rgba(58, 134, 255, 0.1)",
            borderRadius: "8px",
            "&:hover": { backgroundColor: "rgba(58, 134, 255, 0.2)" }
          }}
        >
          <Menu sx={{ color: "#3a86ff" }} />
        </IconButton>
      </Box>
      
      <Divider sx={{ mx: 2, opacity: 0.6 }} />
      
      <List sx={{ mt: 2 }}>
        {menuItems.map(({ text, icon, page }) => (
          <Tooltip title={open ? "" : text} placement="right" key={text}>
            <ListItem disablePadding sx={{ display: "block", my: 0.5 }}>
              <ListItemButton 
                onClick={() => handleMenuClick(page)} 
                selected={activeItem === page}
                sx={{
                  borderRadius: "10px",
                  minHeight: 48,
                  justifyContent: open ? "initial" : "center",
                  mx: 2,
                  px: 2.5,
                  transition: "all 0.2s ease-in-out",
                  "&.Mui-selected": { 
                    backgroundColor: "#3a86ff", 
                    color: "white",
                    "& .MuiListItemIcon-root": { color: "white" }
                  },
                  "&:hover": { 
                    backgroundColor: activeItem === page ? "#3a86ff" : "rgba(58, 134, 255, 0.1)",
                    transform: "translateY(-2px)"
                  }
                }}
              >
                <ListItemIcon 
                  sx={{
                    minWidth: 0,
                    mr: open ? 2 : "auto",
                    justifyContent: "center",
                    color: activeItem === page ? (open ? "white" : "#3a86ff") : "#666"
                  }}
                >
                  {icon}
                </ListItemIcon>
                {open && <ListItemText 
                  primary={text} 
                  sx={{ 
                    opacity: open ? 1 : 0,
                    "& .MuiTypography-root": { 
                      fontWeight: activeItem === page ? 600 : 400,
                      fontSize: "0.9rem"
                    }
                  }} 
                />}
              </ListItemButton>
            </ListItem>
          </Tooltip>
        ))}
      </List>
    </Drawer>
  );
};

export default Sidebar;