import Foundation
@preconcurrency import MLX
import MLXNN
public struct TokenizedText {
    public let tokens: MLXArray
}

public final class SentencePieceTokenizer {
    public let tokenizer: UnigramTokenizer

    // wangqi 2026-02-27: support tokenizer.model (SentencePiece binary) as fallback when tokenizer.json is absent
    public init(nBins: Int, modelFolder: URL) async throws {
        // Try tokenizer.json first (HuggingFace fast tokenizer format)
        let tokenizerJSON = modelFolder.appendingPathComponent("tokenizer.json")
        if FileManager.default.fileExists(atPath: tokenizerJSON.path),
           let data = try? Data(contentsOf: tokenizerJSON),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            self.tokenizer = try UnigramTokenizer(tokenizerJSON: json)
            return
        }

        // Fall back to tokenizer.model (SentencePiece protobuf binary format)
        let tokenizerModel = modelFolder.appendingPathComponent("tokenizer.model")
        guard FileManager.default.fileExists(atPath: tokenizerModel.path),
              let data = try? Data(contentsOf: tokenizerModel) else {
            throw NSError(
                domain: "PocketTTSConditioners",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Missing tokenizer.json or tokenizer.model in \(modelFolder.path)"]
            )
        }
        let (unkId, vocab) = try SentencePieceTokenizer.parseSentencePieceProto(data)
        let json: [String: Any] = ["model": ["unk_id": unkId, "vocab": vocab] as [String: Any]]
        self.tokenizer = try UnigramTokenizer(tokenizerJSON: json)
    }

    // MARK: - SentencePiece Protobuf Parser

    /// Minimal protobuf parser for SentencePiece ModelProto.
    /// Extracts pieces[].piece (string) and pieces[].score (float) to build vocab.
    /// pieces[].type == 2 (UNKNOWN) identifies the unknown token id.
    private static func parseSentencePieceProto(_ data: Data) throws -> (unkId: Int, vocab: [[Any]]) {
        var vocab: [[Any]] = []
        var unkId = 0
        var pos = 0

        func readVarint() throws -> UInt64 {
            var result: UInt64 = 0
            var shift = 0
            while pos < data.count {
                let byte = UInt64(data[pos])
                pos += 1
                result |= (byte & 0x7F) << UInt64(shift)
                if byte & 0x80 == 0 { return result }
                shift += 7
                guard shift < 64 else {
                    throw NSError(domain: "ProtoParser", code: 1, userInfo: [NSLocalizedDescriptionKey: "Varint overflow"])
                }
            }
            throw NSError(domain: "ProtoParser", code: 2, userInfo: [NSLocalizedDescriptionKey: "Unexpected end of data reading varint"])
        }

        func skipField(wireType: UInt64) throws {
            switch wireType {
            case 0:
                _ = try readVarint()
            case 1:
                guard pos + 8 <= data.count else { throw NSError(domain: "ProtoParser", code: 3, userInfo: [:]) }
                pos += 8
            case 2:
                let len = try readVarint()
                guard pos + Int(len) <= data.count else { throw NSError(domain: "ProtoParser", code: 3, userInfo: [:]) }
                pos += Int(len)
            case 5:
                guard pos + 4 <= data.count else { throw NSError(domain: "ProtoParser", code: 3, userInfo: [:]) }
                pos += 4
            default:
                throw NSError(domain: "ProtoParser", code: 4, userInfo: [NSLocalizedDescriptionKey: "Unknown wire type \(wireType)"])
            }
        }

        // Parse top-level ModelProto message
        while pos < data.count {
            let tag = try readVarint()
            let fieldNumber = tag >> 3
            let wireType = tag & 0x7

            guard fieldNumber == 1, wireType == 2 else {
                // Skip non-piece fields (trainer_spec, normalizer_spec, etc.)
                try skipField(wireType: wireType)
                continue
            }

            // Parse SentencePiece sub-message (field 1, length-delimited)
            let msgLen = try readVarint()
            guard pos + Int(msgLen) <= data.count else {
                throw NSError(domain: "ProtoParser", code: 5, userInfo: [NSLocalizedDescriptionKey: "SentencePiece message length out of bounds"])
            }
            let msgEnd = pos + Int(msgLen)

            var piece = ""
            var score: Float = 0.0
            var type_: UInt64 = 1 // NORMAL

            while pos < msgEnd {
                let pieceTag = try readVarint()
                let pfn = pieceTag >> 3
                let pwt = pieceTag & 0x7

                switch (pfn, pwt) {
                case (1, 2): // piece: length-delimited string
                    let slen = try readVarint()
                    guard pos + Int(slen) <= data.count else {
                        throw NSError(domain: "ProtoParser", code: 5, userInfo: [:])
                    }
                    piece = String(bytes: data[pos..<(pos + Int(slen))], encoding: .utf8) ?? ""
                    pos += Int(slen)
                case (2, 5): // score: 32-bit float (little-endian)
                    guard pos + 4 <= data.count else {
                        throw NSError(domain: "ProtoParser", code: 5, userInfo: [:])
                    }
                    let b0 = UInt32(data[pos])
                    let b1 = UInt32(data[pos + 1]) << 8
                    let b2 = UInt32(data[pos + 2]) << 16
                    let b3 = UInt32(data[pos + 3]) << 24
                    score = Float(bitPattern: b0 | b1 | b2 | b3)
                    pos += 4
                case (3, 0): // type: varint enum (NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4, BYTE=6)
                    type_ = try readVarint()
                default:
                    try skipField(wireType: pwt)
                }
            }
            pos = msgEnd

            vocab.append([piece, Double(score)])
            if type_ == 2 { unkId = vocab.count - 1 } // UNKNOWN type
        }

        return (unkId, vocab)
    }

    public func callAsFunction(_ text: String) -> TokenizedText {
        let ids = tokenizer.encodeWithByteFallback(text)
        let arr = MLXArray(ids).expandedDimensions(axis: 0)
        return TokenizedText(tokens: arr)
    }

    public func encode(_ text: String) -> [Int] {
        tokenizer.encodeWithByteFallback(text)
    }

    public func decode(_ ids: [Int]) -> String {
        tokenizer.decode(ids)
    }
}

public final class LUTConditioner: Module {
    public let tokenizer: SentencePieceTokenizer
    public let dim: Int
    public let outputDim: Int

    @ModuleInfo(key: "embed") public var embed: Embedding
    @ModuleInfo(key: "output_proj") public var output_proj: Linear?

    public init(nBins: Int, modelFolder: URL, dim: Int, outputDim: Int) async throws {
        self.tokenizer = try await SentencePieceTokenizer(nBins: nBins, modelFolder: modelFolder)
        self.dim = dim
        self.outputDim = outputDim
        self._embed = ModuleInfo(wrappedValue: Embedding(embeddingCount: nBins + 1, dimensions: dim))
        if dim == outputDim {
            self._output_proj = ModuleInfo(wrappedValue: nil)
        } else {
            self._output_proj = ModuleInfo(wrappedValue: Linear(dim, outputDim, bias: false))
        }
        super.init()
    }

    public func prepare(_ text: String) -> TokenizedText {
        tokenizer(text)
    }

    public func callAsFunction(_ inputs: TokenizedText) -> MLXArray {
        var embeds = embed(inputs.tokens)
        if let proj = output_proj {
            embeds = proj(embeds)
        }
        return embeds
    }
}
