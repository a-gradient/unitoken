import { createRoot } from "react-dom/client";
import Home from "./page";
import "./globals.css";

const root = document.getElementById("root");

if (root === null) {
  throw new Error("Missing FFBPE Inspect root element.");
}

createRoot(root).render(<Home />);
