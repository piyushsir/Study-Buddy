"use client";

import React, { useEffect, useState } from "react";

import Link from "next/link";
import { sourceCodePro } from "./styles/fonts";
import HamburgerMenu from "./components/HamburgerMenu";
const Navbar = () => {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  // const Navbar = () => {
  return (
    <nav className="fixed z-10 top-0 bg-gray-50 text-gray-800 w-full p-4 grid grid-cols-3 items-center">
      <a href="/" className={`text-center`}>
          2-08-3023
      </a>
      {isClient && <HamburgerMenu />}{" "}
      {/* Render HamburgerMenu component on the client side */}
      <p className={`text-center`}></p>
      <div className="hidden">
        <Link href="/">Home 🏡 </Link>
        {/* Projects */}
        {/* <Link href="/page-template">Page Template ©️</Link> */}
        {/* Short Tutorials */}
        {/* <Link href="/chatcompletions">Chat Completions 💬</Link> */}
        <Link href="/pdf">PDF-GPT 👨🏻‍🏫</Link>
        <Link href="/memory">Memory 🧠</Link>
        <Link href="/streaming">Streaming 🌊</Link>

        {/* Documents / QA */}
        <Link href="/transcript-qa">YouTube Video Chat 💬</Link>

        

       
      </div>
    </nav>
  );
};

export default Navbar;
