/// Copyright (c)  2026  Xiaomi Corporation
///                2026  Waad Ben Kheder

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineTtsOmnivoiceModelConfig
    {
        public OfflineTtsOmnivoiceModelConfig()
        {
            Model = "";
            CodecEncoder = "";
            CodecDecoder = "";
            TokenizerDir = "";
            PrefixModel = "";
            TargetModel = "";
            NumSteps = 32;
            TShift = 0.1F;
            GuidanceScale = 2.0F;
            LayerPenaltyFactor = 5.0F;
            PositionTemperature = 5.0F;
            Seed = -1;
        }

        [MarshalAs(UnmanagedType.LPStr)]
        public string Model;

        [MarshalAs(UnmanagedType.LPStr)]
        public string CodecEncoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string CodecDecoder;

        [MarshalAs(UnmanagedType.LPStr)]
        public string TokenizerDir;

        [MarshalAs(UnmanagedType.LPStr)]
        public string PrefixModel;

        [MarshalAs(UnmanagedType.LPStr)]
        public string TargetModel;

        public int NumSteps;
        public float TShift;
        public float GuidanceScale;
        public float LayerPenaltyFactor;
        public float PositionTemperature;
        public int Seed;
    }
}
