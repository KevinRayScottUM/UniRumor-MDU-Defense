import { BrowserRouter, Route, Routes } from "react-router-dom";

import { ApplicationLayout } from "../components/ApplicationLayout";
import { HomePage } from "../pages/HomePage";
import { JobStatusPage } from "../pages/JobStatusPage";
import { ResultPage } from "../pages/ResultPage";

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<ApplicationLayout />}>
        <Route index element={<HomePage />} />
        <Route path="jobs/:jobId" element={<JobStatusPage />} />
        <Route path="jobs/:jobId/result" element={<ResultPage />} />
      </Route>
    </Routes>
  );
}

export function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}

