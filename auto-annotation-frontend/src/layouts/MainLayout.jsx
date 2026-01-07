import React from "react";
import { Outlet } from "react-router-dom";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

export default function MainLayout() {
  return (
    <div className="flex flex-col min-h-screen bg-slate-50 text-slate-900">
      <Navbar className="shadow-md bg-white" />
      <main className="flex-1 container mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <Outlet />
      </main>
      <Footer className="bg-white border-t border-slate-200" />
    </div>
  );
}
