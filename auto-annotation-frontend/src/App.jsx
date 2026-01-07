import React from "react";
import { Routes, Route } from "react-router-dom";

import MainLayout from "./layouts/MainLayout";
import Home from "./pages/Home";
import Upload from "./pages/Upload";
import Projects from "./pages/Projects";
import Project from "./pages/Project";
import Annotate from "./pages/Annotate";

export default function App() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <Routes>
        <Route element={<MainLayout />}>
          <Route index element={<Home />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/annotate/:jobId" element={<Annotate />} />
          <Route path="/projects" element={<Projects />} />
          <Route path="/projects/:id" element={<Project />} />
        </Route>
      </Routes>
    </div>
  );
}
